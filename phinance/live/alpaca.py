"""
phinance.live.alpaca
=====================

Alpaca Markets REST broker adapter.

Supports both paper trading and live trading via the Alpaca REST API.

Prerequisites
-------------
    pip install alpaca-py>=0.20.0

Environment variables
---------------------
    ALPACA_API_KEY     — API key (required)
    ALPACA_SECRET_KEY  — Secret key (required)
    ALPACA_BASE_URL    — https://paper-api.alpaca.markets (paper)
                         https://api.alpaca.markets        (live)
    ALPACA_FEED        — "iex" (free) | "sip" (paid, default: "iex")

References
----------
* https://docs.alpaca.markets/
* https://alpaca-py.readthedocs.io/
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from phinance.live.broker_base import BrokerAdapter, OrderSide, OrderType, OrderStatus
from phinance.live.order_models import Order, Fill, Position
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import — only needed if this module is actually used
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaSide,
        OrderType as AlpacaOT,
        TimeInForce,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestBarRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL  = "https://api.alpaca.markets"


class AlpacaBroker(BrokerAdapter):
    """Alpaca Markets broker adapter.

    Parameters
    ----------
    api_key    : str — Alpaca API key (falls back to ALPACA_API_KEY env var)
    secret_key : str — Alpaca secret key (falls back to ALPACA_SECRET_KEY)
    base_url   : str — API base URL (falls back to ALPACA_BASE_URL)
    feed       : str — market data feed ``"iex"`` | ``"sip"`` (default ``"iex"``)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url:   Optional[str] = None,
        feed:       str = "iex",
    ) -> None:
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. "
                "Install with: pip install alpaca-py"
            )
        self._api_key    = api_key    or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url   = base_url   or os.environ.get("ALPACA_BASE_URL", _PAPER_URL)
        self._feed       = feed or os.environ.get("ALPACA_FEED", "iex")

        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._paper_mode = _PAPER_URL in self._base_url

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper_mode,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        acct = self._trading_client.get_account()
        logger.info(
            "AlpacaBroker connected: account=%s  equity=%.2f",
            acct.id,
            float(acct.equity),
        )

    def disconnect(self) -> None:
        self._trading_client = None
        self._data_client    = None

    # ── Account / market data ─────────────────────────────────────────────────

    def get_account(self) -> Dict:
        if not self._trading_client:
            raise RuntimeError("Not connected — call connect() first")
        acct = self._trading_client.get_account()
        return {
            "id":            str(acct.id),
            "equity":        float(acct.equity),
            "cash":          float(acct.cash),
            "buying_power":  float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "pattern_day_trader": acct.pattern_day_trader,
            "trading_blocked":    acct.trading_blocked,
        }

    def get_positions(self) -> List[Position]:
        if not self._trading_client:
            raise RuntimeError("Not connected")
        raw_positions = self._trading_client.get_all_positions()
        result = []
        for p in raw_positions:
            result.append(Position(
                symbol        = str(p.symbol),
                qty           = float(p.qty),
                avg_entry     = float(p.avg_entry_price),
                market_value  = float(p.market_value) if p.market_value else None,
                unrealised_pl = float(p.unrealized_pl) if p.unrealized_pl else None,
                current_price = float(p.current_price) if p.current_price else None,
            ))
        return result

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        if not self._data_client:
            return None
        try:
            req = StockLatestBarRequest(symbol_or_symbols=[symbol], feed=self._feed)
            bars = self._data_client.get_stock_latest_bar(req)
            bar  = bars.get(symbol)
            if bar is None:
                return None
            return pd.Series({
                "open":   float(bar.open),
                "high":   float(bar.high),
                "low":    float(bar.low),
                "close":  float(bar.close),
                "volume": float(bar.volume),
            })
        except Exception as exc:
            logger.warning("AlpacaBroker.get_latest_bar(%s) failed: %s", symbol, exc)
            return None

    # ── Order management ──────────────────────────────────────────────────────

    def submit_order(
        self,
        symbol:          str,
        qty:             float,
        side:            OrderSide,
        order_type:      OrderType = OrderType.MARKET,
        limit_price:     Optional[float] = None,
        stop_price:      Optional[float] = None,
        time_in_force:   str = "day",
        client_order_id: Optional[str] = None,
    ) -> Order:
        if not self._trading_client:
            raise RuntimeError("Not connected")

        alpaca_side = AlpacaSide.BUY if side in (OrderSide.BUY, "buy") else AlpacaSide.SELL
        tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

        try:
            if order_type in (OrderType.MARKET, "market"):
                req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=alpaca_side, time_in_force=tif,
                    client_order_id=client_order_id,
                )
            elif order_type in (OrderType.LIMIT, "limit"):
                req = LimitOrderRequest(
                    symbol=symbol, qty=qty, side=alpaca_side, time_in_force=tif,
                    limit_price=limit_price, client_order_id=client_order_id,
                )
            else:
                req = StopOrderRequest(
                    symbol=symbol, qty=qty, side=alpaca_side, time_in_force=tif,
                    stop_price=stop_price, client_order_id=client_order_id,
                )
            raw = self._trading_client.submit_order(req)
            return self._to_order(raw)
        except Exception as exc:
            logger.error("AlpacaBroker.submit_order failed: %s", exc)
            return Order(
                order_id=str(client_order_id or "error"),
                symbol=symbol, side=str(side), qty=qty,
                status=OrderStatus.REJECTED.value,
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self._trading_client:
            return False
        try:
            self._trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        if not self._trading_client:
            return None
        try:
            raw = self._trading_client.get_order_by_id(order_id)
            return self._to_order(raw)
        except Exception:
            return None

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        if not self._trading_client:
            return []
        try:
            query_status = QueryOrderStatus.ALL
            if status == "open":
                query_status = QueryOrderStatus.OPEN
            elif status == "closed":
                query_status = QueryOrderStatus.CLOSED
            req = GetOrdersRequest(status=query_status, limit=limit)
            raws = self._trading_client.get_orders(req)
            return [self._to_order(r) for r in raws]
        except Exception as exc:
            logger.warning("list_orders failed: %s", exc)
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_order(raw) -> Order:
        """Convert an Alpaca order object to our Order dataclass."""
        filled_avg = None
        try:
            filled_avg = float(raw.filled_avg_price) if raw.filled_avg_price else None
        except (TypeError, ValueError):
            pass

        return Order(
            order_id          = str(raw.id),
            symbol            = str(raw.symbol),
            side              = str(raw.side.value),
            qty               = float(raw.qty or 0),
            order_type        = str(raw.type.value),
            status            = str(raw.status.value),
            limit_price       = float(raw.limit_price) if raw.limit_price else None,
            stop_price        = float(raw.stop_price)  if raw.stop_price  else None,
            filled_qty        = float(raw.filled_qty or 0),
            filled_avg_price  = filled_avg,
            time_in_force     = str(raw.time_in_force.value),
            client_order_id   = raw.client_order_id,
            submitted_at      = raw.submitted_at,
            filled_at         = raw.filled_at,
        )

    @property
    def is_paper(self) -> bool:
        return self._paper_mode
