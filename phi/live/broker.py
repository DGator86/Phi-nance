"""Broker abstraction and Alpaca adapter for live/paper trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BrokerAccount:
    cash: float
    equity: float
    buying_power: float


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0


@dataclass
class BrokerOrder:
    symbol: str
    qty: float
    side: str
    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    id: str | None = None
    status: str = "new"
    submitted_at: datetime | None = None


class Broker(ABC):
    """Abstract broker interface used by LiveEngine."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_account(self) -> BrokerAccount: ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]: ...

    @abstractmethod
    def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> "pd.DataFrame": ...

    @abstractmethod
    def subscribe_bars(self, symbol: str, callback: Callable[[dict[str, Any]], None]) -> None: ...

    @abstractmethod
    def place_order(self, order: BrokerOrder) -> BrokerOrder: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...


class AlpacaBroker(Broker):
    """Thin Alpaca implementation, with polling fallback for bars."""

    def __init__(self, api_key: str, secret_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self._api = None

    def connect(self) -> None:
        try:
            import alpaca_trade_api as tradeapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("alpaca-trade-api is required for AlpacaBroker") from exc
        self._api = tradeapi.REST(self.api_key, self.secret_key, self.base_url)
        logger.info("Connected to Alpaca base_url=%s", self.base_url)

    def disconnect(self) -> None:
        self._api = None

    def _require_api(self):
        if self._api is None:
            raise RuntimeError("Broker not connected")
        return self._api

    def get_account(self) -> BrokerAccount:
        account = self._require_api().get_account()
        return BrokerAccount(cash=float(account.cash), equity=float(account.equity), buying_power=float(account.buying_power))

    def get_positions(self) -> list[BrokerPosition]:
        positions = self._require_api().list_positions()
        return [
            BrokerPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(getattr(p, "current_price", 0.0) or 0.0),
            )
            for p in positions
        ]

    def get_open_orders(self) -> list[BrokerOrder]:
        orders = self._require_api().list_orders(status="open")
        return [
            BrokerOrder(
                symbol=o.symbol,
                qty=float(o.qty),
                side=str(o.side),
                order_type=str(o.type),
                limit_price=float(o.limit_price) if getattr(o, "limit_price", None) else None,
                id=o.id,
                status=str(o.status),
            )
            for o in orders
        ]

    def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str):
        bars = self._require_api().get_bars(symbol, timeframe, start.isoformat(), end.isoformat()).df
        if bars.empty:
            return bars
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol)
        return bars

    def subscribe_bars(self, symbol: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Polling fallback; engine calls this every update cycle."""
        end = datetime.utcnow()
        from datetime import timedelta
        start = end - timedelta(minutes=5)
        bars = self.get_historical_bars(symbol, start=start, end=end, timeframe="1Min")
        if bars.empty:
            return
        last = bars.iloc[-1]
        callback({"symbol": symbol, "timestamp": bars.index[-1], "open": float(last.open), "high": float(last.high), "low": float(last.low), "close": float(last.close), "volume": float(last.volume)})

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        o = self._require_api().submit_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            type=order.order_type,
            time_in_force="day",
            limit_price=order.limit_price,
            stop_price=order.stop_price,
        )
        order.id = o.id
        order.status = str(o.status)
        order.submitted_at = datetime.utcnow()
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._require_api().cancel_order(order_id)
        return True
