"""
phinance.live.order_models
===========================

Lightweight dataclasses for orders, fills and positions.

These are broker-agnostic and used by all adapters to provide a
uniform data model throughout the live trading layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    """A trading order (pending, submitted, or filled)."""

    order_id:        str
    symbol:          str
    side:            str         # "buy" | "sell"
    qty:             float
    order_type:      str = "market"   # "market" | "limit" | "stop"
    status:          str = "pending"  # "pending" | "submitted" | "filled" | "cancelled" | "rejected"
    limit_price:     Optional[float] = None
    stop_price:      Optional[float] = None
    filled_qty:      float = 0.0
    filled_avg_price: Optional[float] = None
    time_in_force:   str = "day"
    client_order_id: Optional[str] = None
    submitted_at:    Optional[datetime] = None
    filled_at:       Optional[datetime] = None

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_pending(self) -> bool:
        return self.status in ("pending", "submitted")

    def __repr__(self) -> str:
        return (
            f"Order({self.symbol} {self.side} {self.qty}@{self.order_type} "
            f"status={self.status})"
        )


@dataclass
class Fill:
    """A single trade execution (partial or full fill)."""

    order_id:   str
    symbol:     str
    side:       str
    qty:        float
    price:      float
    filled_at:  datetime = field(default_factory=datetime.utcnow)
    commission: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    """An open position in a specific symbol."""

    symbol:       str
    qty:          float          # positive = long, negative = short
    avg_entry:    float          # average entry price
    market_value: Optional[float] = None
    unrealised_pl: Optional[float] = None
    current_price: Optional[float] = None

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def __repr__(self) -> str:
        return f"Position({self.symbol} qty={self.qty} @ {self.avg_entry:.2f})"
