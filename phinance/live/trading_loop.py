"""
phinance.live.trading_loop
===========================

LiveTradingLoop — the main orchestrator for live / paper trading.

Architecture
------------
The loop:
  1. Fetches the latest market bar from the broker adapter.
  2. Appends it to a rolling OHLCV buffer.
  3. Computes the composite signal using the configured indicators + blender.
  4. Passes the signal to a rule-based decision engine.
  5. Submits buy/sell/hold orders via the broker adapter.
  6. Logs all activity and stores trade records.

The loop is designed to be called periodically (e.g. by a scheduler or
Streamlit rerun) rather than blocking in an infinite loop — this makes
it easy to integrate with both async frameworks and Streamlit's
session-state model.

Usage
-----
    from phinance.live import LiveTradingLoop, PaperBroker
    from phinance.live.broker_base import OrderSide

    broker = PaperBroker(initial_capital=50_000)
    broker.connect()
    broker.update_price("SPY", 450.0)

    loop = LiveTradingLoop(
        broker     = broker,
        symbol     = "SPY",
        indicators = {"RSI": {"enabled": True, "params": {"period": 14}}},
        capital    = 50_000,
    )
    result = loop.run_once(ohlcv_df)
    print(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.live.broker_base import BrokerAdapter, OrderSide, OrderType
from phinance.live.order_models import Order, Position
from phinance.blending.blender import blend_signals
from phinance.strategies.indicator_catalog import compute_indicator
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LiveRunResult:
    """Summary of a single live trading loop iteration."""

    timestamp:     datetime
    symbol:        str
    signal:        float           # composite signal value
    action:        str             # "buy" | "sell" | "hold"
    reason:        str             # human-readable rationale
    order:         Optional[Order] = None
    account:       Optional[Dict]  = None
    positions:     List[Position]  = field(default_factory=list)
    error:         Optional[str]   = None

    def __str__(self) -> str:
        ts   = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        base = f"[{ts}] {self.symbol} | signal={self.signal:+.3f} | {self.action} | {self.reason}"
        if self.order:
            base += f" | order={self.order}"
        if self.error:
            base += f" | ERROR: {self.error}"
        return base


class LiveTradingLoop:
    """Phi-nance live / paper trading orchestrator.

    Parameters
    ----------
    broker         : BrokerAdapter — connected broker (Alpaca, IBKR, or Paper)
    symbol         : str           — ticker to trade (e.g. ``"SPY"``)
    indicators     : dict          — ``{name: {\"enabled\": bool, \"params\": dict}}``
    blend_method   : str           — signal blend method (default ``"weighted_sum"``)
    blend_weights  : dict, optional — ``{indicator_name: weight}``
    signal_threshold : float       — minimum |signal| to trigger a trade (default 0.15)
    position_size_pct : float      — fraction of available cash to deploy (default 0.90)
    capital        : float         — initial/reference capital for sizing (default 100 000)
    max_buffer_bars : int          — rolling OHLCV buffer length (default 500)
    """

    def __init__(
        self,
        broker:            BrokerAdapter,
        symbol:            str = "SPY",
        indicators:        Optional[Dict[str, Dict[str, Any]]] = None,
        blend_method:      str = "weighted_sum",
        blend_weights:     Optional[Dict[str, float]] = None,
        signal_threshold:  float = 0.15,
        position_size_pct: float = 0.90,
        capital:           float = 100_000.0,
        max_buffer_bars:   int   = 500,
    ) -> None:
        self.broker             = broker
        self.symbol             = symbol
        self.indicators         = indicators or {"RSI": {"enabled": True, "params": {}}}
        self.blend_method       = blend_method
        self.blend_weights      = blend_weights
        self.signal_threshold   = signal_threshold
        self.position_size_pct  = position_size_pct
        self.capital            = capital
        self.max_buffer_bars    = max_buffer_bars

        self._ohlcv_buffer: Optional[pd.DataFrame] = None
        self._last_signal:  float = 0.0
        self._run_log: List[LiveRunResult] = []

    # ── Buffer management ─────────────────────────────────────────────────────

    def seed_buffer(self, historical_ohlcv: pd.DataFrame) -> None:
        """Seed the rolling OHLCV buffer with historical data.

        Call this once before the first ``run_once()`` to give indicators
        enough warm-up data.

        Parameters
        ----------
        historical_ohlcv : pd.DataFrame — recent OHLCV (≤ max_buffer_bars rows)
        """
        self._ohlcv_buffer = historical_ohlcv.tail(self.max_buffer_bars).copy()
        logger.info(
            "LiveTradingLoop buffer seeded: %d bars for %s",
            len(self._ohlcv_buffer), self.symbol,
        )

    def _append_bar(self, bar: pd.Series, timestamp: datetime) -> None:
        """Append a new bar to the rolling buffer."""
        new_row = pd.DataFrame([bar], index=[pd.Timestamp(timestamp)])
        if self._ohlcv_buffer is None:
            self._ohlcv_buffer = new_row
        else:
            self._ohlcv_buffer = pd.concat([self._ohlcv_buffer, new_row]).tail(
                self.max_buffer_bars
            )

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_signal(self, ohlcv: pd.DataFrame) -> float:
        """Compute composite signal from the current OHLCV buffer."""
        active = {
            name: cfg for name, cfg in self.indicators.items()
            if cfg.get("enabled", False)
        }
        if not active:
            return 0.0

        signals_dict: Dict[str, pd.Series] = {}
        for name, cfg in active.items():
            try:
                sig = compute_indicator(name, ohlcv, cfg.get("params", {}))
                signals_dict[name] = sig
            except Exception as exc:
                logger.warning("Signal compute failed for %s: %s", name, exc)

        if not signals_dict:
            return 0.0

        signals_df = pd.DataFrame(signals_dict)
        composite  = blend_signals(
            signals_df,
            weights=self.blend_weights,
            method=self.blend_method,
        )
        return float(composite.iloc[-1]) if len(composite) > 0 else 0.0

    # ── Decision logic ────────────────────────────────────────────────────────

    def _decide(
        self,
        signal:    float,
        positions: List[Position],
    ) -> tuple[str, str]:
        """Translate a signal into buy/sell/hold with rationale.

        Returns
        -------
        (action, reason)
        """
        is_long = any(p.symbol == self.symbol and p.qty > 0 for p in positions)

        if signal > self.signal_threshold and not is_long:
            return "buy", f"signal={signal:+.3f} > threshold={self.signal_threshold}"
        if signal < -self.signal_threshold and is_long:
            return "sell", f"signal={signal:+.3f} < threshold={-self.signal_threshold}"
        if is_long:
            return "hold", f"long position held, signal={signal:+.3f}"
        return "hold", f"flat, signal={signal:+.3f} within threshold"

    # ── Main loop entry point ─────────────────────────────────────────────────

    def run_once(
        self,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> LiveRunResult:
        """Execute one iteration of the live trading loop.

        Parameters
        ----------
        ohlcv : pd.DataFrame, optional
            If provided, append the last row to the rolling buffer.
            If None, the loop tries to fetch the latest bar from the broker.

        Returns
        -------
        LiveRunResult
        """
        ts     = datetime.utcnow()
        result = LiveRunResult(timestamp=ts, symbol=self.symbol, signal=0.0,
                               action="hold", reason="initialising")

        try:
            # ── Step 1: Update OHLCV buffer ───────────────────────────────────
            if ohlcv is not None:
                if len(ohlcv) > 0:
                    last_bar = ohlcv.iloc[-1]
                    self._append_bar(last_bar, ohlcv.index[-1])
                    if self._ohlcv_buffer is None or len(self._ohlcv_buffer) < len(ohlcv):
                        self._ohlcv_buffer = ohlcv.tail(self.max_buffer_bars).copy()
            else:
                bar = self.broker.get_latest_bar(self.symbol)
                if bar is not None:
                    self._append_bar(bar, ts)

            if self._ohlcv_buffer is None or len(self._ohlcv_buffer) < 10:
                result.reason = "insufficient data in buffer"
                return result

            # ── Step 2: Compute composite signal ──────────────────────────────
            signal = self._compute_signal(self._ohlcv_buffer)
            self._last_signal = signal
            result.signal = signal

            # ── Step 3: Get account state ──────────────────────────────────────
            try:
                account   = self.broker.get_account()
                positions = self.broker.get_positions()
                result.account   = account
                result.positions = positions
            except Exception as exc:
                logger.warning("Could not fetch account/positions: %s", exc)
                account   = {}
                positions = []

            # ── Step 4: Decide ────────────────────────────────────────────────
            action, reason = self._decide(signal, positions)
            result.action = action
            result.reason = reason

            # ── Step 5: Submit order ──────────────────────────────────────────
            if action in ("buy", "sell"):
                cash       = float(account.get("cash", self.capital))
                last_price = float(
                    (self._ohlcv_buffer["close"].iloc[-1])
                    if "close" in self._ohlcv_buffer.columns
                    else 1.0
                )
                if last_price <= 0:
                    last_price = 1.0

                if action == "buy":
                    qty = int(cash * self.position_size_pct / last_price)
                    if qty > 0:
                        order = self.broker.submit_order(
                            self.symbol, qty, OrderSide.BUY
                        )
                        result.order = order
                        logger.info("BUY %d %s @ ~%.2f", qty, self.symbol, last_price)
                else:
                    # Sell entire position
                    pos_qty = sum(
                        p.qty for p in positions
                        if p.symbol == self.symbol and p.qty > 0
                    )
                    if pos_qty > 0:
                        order = self.broker.submit_order(
                            self.symbol, pos_qty, OrderSide.SELL
                        )
                        result.order = order
                        logger.info("SELL %d %s @ ~%.2f", pos_qty, self.symbol, last_price)

        except Exception as exc:
            result.error = str(exc)
            logger.error("LiveTradingLoop.run_once error: %s", exc, exc_info=True)

        self._run_log.append(result)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_run_log(self) -> List[LiveRunResult]:
        """Return the history of all loop iterations."""
        return list(self._run_log)

    def run_loop(
        self,
        ohlcv:        pd.DataFrame,
        interval_sec: float = 60.0,
        max_iterations: int = 0,
    ) -> List[LiveRunResult]:
        """Run the trading loop continuously (blocking).

        Parameters
        ----------
        ohlcv          : pd.DataFrame — initial historical data
        interval_sec   : float        — seconds between iterations (default 60)
        max_iterations : int          — stop after N iterations; 0 = run forever

        Returns
        -------
        List of LiveRunResult (only populated when max_iterations > 0)
        """
        self.seed_buffer(ohlcv)
        iteration = 0
        results: List[LiveRunResult] = []

        while True:
            res = self.run_once()
            results.append(res)
            logger.info(str(res))

            iteration += 1
            if max_iterations > 0 and iteration >= max_iterations:
                break

            time.sleep(interval_sec)

        return results
