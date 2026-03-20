"""Real-time portfolio tracker for live trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class PositionState:
    qty: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0


class LivePortfolio:
    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, PositionState] = {}
        self.equity_curve: list[dict[str, float | str]] = []

    def apply_fill(self, symbol: str, qty: float, price: float, side: str) -> None:
        signed_qty = float(qty) if side.lower() == "buy" else -float(qty)
        pos = self.positions.setdefault(symbol, PositionState())
        new_qty = pos.qty + signed_qty
        if signed_qty > 0:
            gross = pos.qty * pos.avg_price + signed_qty * price
            pos.avg_price = gross / new_qty if new_qty else 0.0
        elif new_qty == 0:
            pos.avg_price = 0.0
        pos.qty = new_qty
        pos.last_price = price
        self.cash -= signed_qty * price

    def update_price(self, symbol: str, price: float) -> None:
        pos = self.positions.setdefault(symbol, PositionState())
        pos.last_price = float(price)

    def equity(self) -> float:
        mv = sum(p.qty * (p.last_price or p.avg_price) for p in self.positions.values())
        return self.cash + mv

    def record(self) -> None:
        self.equity_curve.append({"timestamp": datetime.utcnow().isoformat(), "equity": self.equity()})

    def snapshot_positions(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for symbol, pos in self.positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            pnl = (pos.last_price - pos.avg_price) * pos.qty
            rows.append({"symbol": symbol, "qty": pos.qty, "avg_price": pos.avg_price, "current_price": pos.last_price, "pnl": pnl})
        return rows
