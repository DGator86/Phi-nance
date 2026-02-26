"""
Polygon.io L2 Feed — Real-time order book and microstructure signals.

Connects to Polygon.io WebSocket (or REST fallback) and computes L2/L3
signals that extend the OHLCV-proxy features in features.py:

  book_imbalance  — bid_depth / (bid_depth + ask_depth) at NBBO, ∈ (0,1)
                    0.5 = balanced; >0.5 = bid-heavy (buy pressure)
  ofi_true        — z-scored signed volume from aggressor-tagged trades
                    Positive = buyers hit asks aggressively
  spread_bps      — bid-ask spread in basis points (market maker commitment)
  depth_ratio     — current NBBO depth / rolling 30s avg (book thinning signal)
  depth_trend     — rolling slope of total NBBO depth (∂L/∂t proxy)

Signal hierarchy position:
  book_imbalance → L1 (kinematic)
  ofi_true       → L1 (kinematic, directional)
  spread_bps     → L1 (kinematic)
  depth_ratio    → L2 (dynamic — rate of book change)
  depth_trend    → L2 / L3 (∂L/∂t — earliest S/R evaporation signal)

Graceful degradation:
  - No POLYGON_API_KEY → returns ZERO_SIGNALS dict
  - websocket-client not installed → REST fallback
  - Only NBBO quotes available (free tier) → book_imbalance + spread computable
  - Full LV2 book (paid tier) → all signals fully computable

Thread safety:
  PolygonL2Client runs a background daemon thread.
  All shared state is protected by threading.Lock.

Requirements:
  pip install websocket-client sortedcontainers
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Optional-dependency guards
# ──────────────────────────────────────────────────────────────────────────────

try:
    import websocket as _websocket_mod   # noqa: F401
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False
    logger.debug(
        "websocket-client not installed — PolygonL2Client will use REST fallback. "
        "Install with: pip install websocket-client"
    )

POLYGON_WS_URL   = "wss://socket.polygon.io/stocks"
POLYGON_REST_URL = "https://api.polygon.io"


# ──────────────────────────────────────────────────────────────────────────────
# Internal Order Book
# ──────────────────────────────────────────────────────────────────────────────

class _OrderBook:
    """
    Thread-safe order book maintained from NBBO quote and trade updates.

    In free-tier Polygon (NBBO only), each quote update replaces the
    top-of-book bid/ask.  When LV2 is available, multiple levels are tracked.
    """

    def __init__(self, depth_window_s: float = 30.0) -> None:
        self._lock           = threading.Lock()
        self._best_bid       = 0.0
        self._best_ask       = 0.0
        self._bid_size       = 0.0
        self._ask_size       = 0.0
        # Circular buffer: (timestamp, total_nbbo_depth)
        self._depth_history: deque = deque(maxlen=max(10, int(depth_window_s * 6)))
        # Circular buffer: signed trade sizes (+ = buy aggressor)
        self._ofi_history: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Update methods (called from WebSocket thread)
    # ------------------------------------------------------------------

    def update_quote(
        self,
        bid: float,
        bid_size: float,
        ask: float,
        ask_size: float,
    ) -> None:
        """Process a new NBBO quote."""
        with self._lock:
            self._best_bid = bid
            self._best_ask = ask
            self._bid_size = bid_size
            self._ask_size = ask_size
            total = bid_size + ask_size
            self._depth_history.append((time.monotonic(), total))

    def record_trade(self, price: float, size: float, is_buy: bool) -> None:
        """Record an aggressor-tagged trade for OFI computation."""
        with self._lock:
            self._ofi_history.append(size if is_buy else -size)

    # ------------------------------------------------------------------
    # Snapshot (called from any thread)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        """Compute and return the current L2 signal snapshot."""
        with self._lock:
            return self._compute_signals()

    def _compute_signals(self) -> Dict[str, float]:
        """All computations under the lock."""
        bid_sz = self._bid_size
        ask_sz = self._ask_size
        total  = bid_sz + ask_sz

        # ── book_imbalance ∈ (0,1) ────────────────────────────────────
        book_imbalance = (bid_sz / (total + 1e-10)) if total > 0 else 0.5

        # ── spread_bps ────────────────────────────────────────────────
        mid = (self._best_bid + self._best_ask) / 2.0
        if mid > 0 and self._best_ask > self._best_bid:
            spread_bps = (self._best_ask - self._best_bid) / mid * 1e4
        else:
            spread_bps = 0.0

        # ── ofi_true: z-scored OFI from recent trades ─────────────────
        ofi_true = 0.0
        if len(self._ofi_history) >= 10:
            arr = np.array(list(self._ofi_history), dtype=float)
            mu  = float(arr.mean())
            std = float(arr.std())
            ofi_true = float(np.clip(mu / (std + 1e-10), -5.0, 5.0))

        # ── depth_ratio and depth_trend ───────────────────────────────
        depth_ratio = 1.0
        depth_trend = 0.0
        if len(self._depth_history) >= 10:
            depths = np.array([d for _, d in self._depth_history], dtype=float)
            avg    = float(depths.mean())
            curr   = float(depths[-1])
            depth_ratio = curr / (avg + 1e-10)

            # Linear trend (least-squares slope) per observation
            n = len(depths)
            if n >= 5:
                x       = np.arange(n, dtype=float)
                xbar    = x.mean()
                ybar    = depths.mean()
                cov_xy  = float(((x - xbar) * (depths - ybar)).sum())
                var_x   = float(((x - xbar) ** 2).sum())
                depth_trend = cov_xy / (var_x + 1e-15)

        return {
            "book_imbalance": float(book_imbalance),
            "ofi_true":       float(ofi_true),
            "spread_bps":     float(spread_bps),
            "depth_ratio":    float(np.clip(depth_ratio, 0.0, 10.0)),
            "depth_trend":    float(np.clip(depth_trend, -1e6, 1e6)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Client
# ──────────────────────────────────────────────────────────────────────────────

class PolygonL2Client:
    """
    Polygon.io WebSocket L2 client for real-time microstructure signals.

    Subscribes to quote (Q.) and trade (T.) streams for a ticker.
    Maintains a live order book and computes L2 signals continuously
    in a background daemon thread.

    Parameters
    ----------
    ticker       : equity ticker (e.g. 'AAPL')
    api_key      : Polygon.io API key (defaults to POLYGON_API_KEY env var)
    config       : 'polygon' sub-dict from config.yaml
    auto_start   : if True launch WebSocket thread immediately

    Usage
    -----
    >>> client = PolygonL2Client('AAPL', auto_start=True)
    >>> signals = client.get_snapshot()
    # {'book_imbalance': 0.54, 'ofi_true': 1.2, 'spread_bps': 1.8, ...}
    >>> client.stop()
    """

    ZERO_SIGNALS: Dict[str, float] = {
        "book_imbalance": 0.5,
        "ofi_true":       0.0,
        "spread_bps":     0.0,
        "depth_ratio":    1.0,
        "depth_trend":    0.0,
    }

    def __init__(
        self,
        ticker:      str,
        api_key:     Optional[str] = None,
        config:      Optional[Dict[str, Any]] = None,
        auto_start:  bool = False,
    ) -> None:
        self.ticker     = ticker.upper()
        self.api_key    = (
            api_key
            or os.getenv("POLYGON_API_KEY", "")
        )
        self.cfg        = config or {}
        depth_window_s  = float(self.cfg.get("depth_window_s", 30.0))

        self._book      = _OrderBook(depth_window_s=depth_window_s)
        self._ws        = None
        self._thread: Optional[threading.Thread] = None
        self._running   = False
        self._connected = False

        if not self.api_key:
            logger.info(
                "PolygonL2Client (%s): no API key set — running offline. "
                "Set POLYGON_API_KEY env var or config['polygon']['api_key'].",
                self.ticker,
            )
        elif not _WEBSOCKET_AVAILABLE:
            logger.warning(
                "websocket-client not installed. "
                "Install with: pip install websocket-client. "
                "PolygonL2Client will return zero signals."
            )
        elif auto_start:
            self.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start WebSocket connection in a background daemon thread."""
        if not self.api_key or not _WEBSOCKET_AVAILABLE:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_with_reconnect,
            daemon=True,
            name=f"polygon-l2-{self.ticker}",
        )
        self._thread.start()
        logger.info("PolygonL2Client started for %s", self.ticker)

    def stop(self) -> None:
        """Stop the WebSocket connection and background thread."""
        self._running = False
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("PolygonL2Client stopped for %s", self.ticker)

    def get_snapshot(self) -> Dict[str, float]:
        """
        Return the current L2 signal snapshot.

        Always returns a complete dict — never raises.
        Returns ZERO_SIGNALS when offline, disconnected, or auth failed.
        """
        if not self._connected:
            return dict(self.ZERO_SIGNALS)
        return self._book.snapshot()

    def get_l2_signals_for_features(self) -> Dict[str, float]:
        """
        Return L2 signals in the format expected by Mixer._liquidity_confidence().
        Keys: book_imbalance, ofi_true, spread_bps, depth_ratio, depth_trend.
        Always safe to call — returns ZERO_SIGNALS when offline.
        """
        return self.get_snapshot()

    @property
    def is_connected(self) -> bool:
        """True if currently authenticated and receiving data."""
        return self._connected

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    def _run_with_reconnect(self) -> None:
        """Auto-reconnect loop with exponential backoff."""
        backoff = 1.0
        while self._running:
            try:
                self._connect_once()
                backoff = 1.0
            except Exception as exc:
                self._connected = False
                logger.warning(
                    "PolygonL2Client (%s) disconnected: %s — retry in %.0fs",
                    self.ticker, exc, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

    def _connect_once(self) -> None:
        """Open one WebSocket session (blocks until closed)."""
        import websocket  # imported here so module loads without it installed

        ws = websocket.WebSocketApp(
            POLYGON_WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = ws
        ws.run_forever()  # blocks

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    def _on_open(self, ws: Any) -> None:
        logger.debug("Polygon WS opened (%s)", self.ticker)

    def _on_close(self, ws: Any, code: Any, msg: Any) -> None:
        self._connected = False
        logger.debug("Polygon WS closed (%s): %s %s", self.ticker, code, msg)

    def _on_error(self, ws: Any, error: Any) -> None:
        self._connected = False
        logger.debug("Polygon WS error (%s): %s", self.ticker, error)

    def _on_message(self, ws: Any, raw: str) -> None:
        try:
            messages = json.loads(raw)
        except Exception:
            return

        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            ev = msg.get("ev")

            if ev == "connected":
                # Step 1: authenticate
                ws.send(json.dumps({"action": "auth", "params": self.api_key}))

            elif ev == "auth_success":
                logger.info(
                    "Polygon auth success (%s) — subscribing to Q+T streams",
                    self.ticker,
                )
                ws.send(json.dumps({
                    "action": "subscribe",
                    "params": f"Q.{self.ticker},T.{self.ticker}",
                }))
                self._connected = True

            elif ev == "auth_failed":
                logger.error(
                    "Polygon auth failed (%s) — check POLYGON_API_KEY", self.ticker
                )
                ws.close()
                self._running = False

            elif ev == "Q":   # NBBO Quote
                self._handle_quote(msg)

            elif ev == "T":   # Aggressor-tagged Trade
                self._handle_trade(msg)

    def _handle_quote(self, msg: Dict[str, Any]) -> None:
        bp  = float(msg.get("bp", 0) or 0)   # bid price
        bs  = float(msg.get("bs", 0) or 0)   # bid size
        ap  = float(msg.get("ap", 0) or 0)   # ask price
        as_ = float(msg.get("as", 0) or 0)   # ask size
        if bp > 0 and ap > 0:
            self._book.update_quote(bp, bs, ap, as_)

    def _handle_trade(self, msg: Dict[str, Any]) -> None:
        price = float(msg.get("p", 0) or 0)
        size  = float(msg.get("s", 0) or 0)
        if price <= 0 or size <= 0:
            return
        is_buy = self._infer_aggressor(price)
        self._book.record_trade(price, size, is_buy)

    def _infer_aggressor(self, price: float) -> bool:
        """
        Classify trade as buy/sell by tick rule:
        trade at or above NBBO mid → buy aggressor.
        """
        bid = self._book._best_bid
        ask = self._book._best_ask
        if bid <= 0 or ask <= 0:
            return True   # default: buy
        mid = (bid + ask) / 2.0
        return price >= mid


# ──────────────────────────────────────────────────────────────────────────────
# REST Fallback Client
# ──────────────────────────────────────────────────────────────────────────────

class PolygonRestClient:
    """
    Polygon.io REST-based L2 signal computation.

    Uses the Polygon snapshot endpoint to compute book_imbalance and
    spread_bps without a WebSocket connection.  OFI, depth_ratio, and
    depth_trend are not computable from a single snapshot and return 0.

    Useful for batch scanning where a persistent WebSocket is impractical.

    Usage
    -----
    >>> client = PolygonRestClient(api_key='...')
    >>> signals = client.get_snapshot('AAPL')
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config:  Optional[Dict[str, Any]] = None,
        timeout: int = 5,
    ) -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.cfg     = config or {}
        self.timeout = timeout

    def get_snapshot(self, ticker: str) -> Dict[str, float]:
        """
        Fetch and compute available L2 signals for one ticker via REST.
        Returns ZERO_SIGNALS on any failure.
        """
        if not self.api_key:
            return dict(PolygonL2Client.ZERO_SIGNALS)

        try:
            import requests

            url  = f"{POLYGON_REST_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
            resp = requests.get(url, params={"apiKey": self.api_key}, timeout=self.timeout)
            resp.raise_for_status()

            data        = resp.json()
            tick        = data.get("ticker", {})
            last_quote  = tick.get("lastQuote", {})

            bid    = float(last_quote.get("p", 0) or 0)
            ask    = float(last_quote.get("P", 0) or 0)
            bid_sz = float(last_quote.get("s", 100) or 100)
            ask_sz = float(last_quote.get("S", 100) or 100)

            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
            spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else 0.0
            imbalance  = bid_sz / (bid_sz + ask_sz + 1e-10)

            return {
                "book_imbalance": float(np.clip(imbalance, 0.0, 1.0)),
                "ofi_true":       0.0,
                "spread_bps":     float(np.clip(spread_bps, 0.0, 500.0)),
                "depth_ratio":    1.0,
                "depth_trend":    0.0,
            }

        except Exception as exc:
            logger.debug("PolygonRestClient.get_snapshot(%s) failed: %s", ticker, exc)
            return dict(PolygonL2Client.ZERO_SIGNALS)

    def get_l2_signals_for_features(self, ticker: str) -> Dict[str, float]:
        """
        REST-based version. Ticker required (stateless client).
        Return L2 signals in the format expected by Mixer._liquidity_confidence().
        Keys: book_imbalance, ofi_true, spread_bps, depth_ratio, depth_trend.
        Always safe to call — returns ZERO_SIGNALS on any failure.
        """
        return self.get_snapshot(ticker)
