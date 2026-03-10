"""Order models and execution records for LOB simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class OrderSide(str, Enum):
    """Side of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    CANCEL = "cancel"


@dataclass(slots=True)
class SimOrder:
    """Order submitted by a strategy to the simulation engine."""

    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    order_id: str | None = None
    symbol: str = "SIM"
    timestamp: pd.Timestamp | datetime | None = None


@dataclass(slots=True)
class Fill:
    """Execution record generated after order matching."""

    side: OrderSide
    quantity: float
    price: float
    symbol: str = "SIM"
    order_id: str | None = None
    timestamp: pd.Timestamp | datetime | None = None
    notional: float = field(init=False)

    def __post_init__(self) -> None:
        self.notional = float(self.quantity) * float(self.price)
