"""Portfolio accounting helpers for multi-asset backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Order:
    """Simple order model used by :class:`Portfolio`."""

    symbol: str
    shares: float
    price: float
    timestamp: datetime | pd.Timestamp | None = None


class Portfolio:
    """Track cash, positions, transactions, and equity over time."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}
        self.transactions: list[dict[str, float | str | datetime | pd.Timestamp | None]] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update latest known prices for mark-to-market valuation."""
        for symbol, price in prices.items():
            if price is None:
                continue
            p = float(price)
            if p > 0:
                self.current_prices[symbol] = p

    def execute_order(self, order: Order) -> None:
        """Execute an order immediately at the provided price."""
        if order.price <= 0 or order.shares == 0:
            return
        cost = float(order.shares) * float(order.price)
        self.cash -= cost
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + float(order.shares)
        if abs(self.positions[order.symbol]) < 1e-12:
            self.positions.pop(order.symbol, None)
        self.transactions.append(
            {
                "timestamp": order.timestamp,
                "symbol": order.symbol,
                "shares": float(order.shares),
                "price": float(order.price),
                "notional": cost,
            }
        )

    def total_value(self) -> float:
        """Return cash plus mark-to-market value of all positions."""
        value = self.cash
        for symbol, shares in self.positions.items():
            value += shares * self.current_prices.get(symbol, 0.0)
        return float(value)

    def record_equity(self, ts: pd.Timestamp) -> float:
        """Append current total value to equity curve and return it."""
        total = self.total_value()
        self.equity_curve.append((pd.Timestamp(ts), total))
        return total

    def returns(self) -> pd.Series:
        """Compute simple returns from the equity curve."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx = [x[0] for x in self.equity_curve]
        values = [x[1] for x in self.equity_curve]
        return pd.Series(values, index=idx, dtype=float).pct_change().fillna(0.0)

