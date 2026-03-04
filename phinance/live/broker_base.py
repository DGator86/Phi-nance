"""
phinance.live.broker_base
==========================

Abstract broker adapter interface.

All broker implementations (Alpaca, IBKR, Paper) must subclass
``BrokerAdapter`` and implement the abstract methods.  This ensures the
``LiveTradingLoop`` can be tested and swapped without changing any
trading logic.

Design principles
-----------------
* **No side effects at import time** — connecting to the broker is
  deferred to ``connect()`` / ``__enter__``
* **Sync first** — all methods are synchronous; async wrappers can be
  added by callers
* **Idempotent** — calling ``connect()`` on an already-connected adapter
  should be safe
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from phinance.live.order_models import Order, Fill, Position


# ── Enums ──────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


class OrderStatus(str, Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


# ── Abstract Adapter ──────────────────────────────────────────────────────────


class BrokerAdapter(ABC):
    """Abstract base class for all broker adapters.

    Concrete implementations: ``PaperBroker``, ``AlpacaBroker``, ``IBKRBroker``.

    Usage
    -----
        with AlpacaBroker(api_key=..., secret_key=...) as broker:
            broker.submit_order("SPY", 10, OrderSide.BUY)
            print(broker.get_positions())
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker API / gateway."""
        ...

    def disconnect(self) -> None:
        """Cleanly close the broker connection (override if needed)."""

    def __enter__(self) -> "BrokerAdapter":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Account / market data ─────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> Dict:
        """Return account details: equity, cash, buying_power, etc."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Return a list of currently open positions."""
        ...

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Return the most recent OHLCV bar for *symbol*.

        Returns a pd.Series with keys: open, high, low, close, volume.
        Returns None if unavailable.
        """
        ...

    # ── Order management ──────────────────────────────────────────────────────

    @abstractmethod
    def submit_order(
        self,
        symbol:     str,
        qty:        float,
        side:       OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Submit an order.

        Returns
        -------
        Order — with status SUBMITTED or FILLED (for market orders on paper)
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Returns True if cancelled successfully, False otherwise.
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch a specific order by ID."""
        ...

    @abstractmethod
    def list_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Order]:
        """List recent orders."""
        ...

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable broker name."""
        return self.__class__.__name__

    @property
    def is_paper(self) -> bool:
        """True if this is a paper-trading (simulated) adapter."""
        return False

    def __repr__(self) -> str:
        return f"{self.name}(paper={self.is_paper})"
