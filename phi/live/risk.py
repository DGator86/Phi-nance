"""Risk controls for live order validation."""

from __future__ import annotations

from dataclasses import dataclass

from phi.live.broker import BrokerOrder


@dataclass
class RiskLimits:
    max_position_pct: float = 0.25
    max_drawdown_pct: float = 0.2
    daily_loss_limit_pct: float = 0.05


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.paused = False

    def update_equity(self, equity: float) -> None:
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.day_start_equity is None:
            self.day_start_equity = equity
        drawdown = 0.0 if self.peak_equity <= 0 else (self.peak_equity - equity) / self.peak_equity
        daily_loss = 0.0 if self.day_start_equity <= 0 else (self.day_start_equity - equity) / self.day_start_equity
        self.paused = drawdown >= self.limits.max_drawdown_pct or daily_loss >= self.limits.daily_loss_limit_pct

    def validate_order(self, order: BrokerOrder, equity: float, price: float) -> bool:
        if self.paused or order.qty <= 0 or price <= 0:
            return False
        notional = abs(order.qty * price)
        return (notional / max(equity, 1e-9)) <= self.limits.max_position_pct
