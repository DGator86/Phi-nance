"""Simple portfolio simulator used by RL training environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PortfolioSimulator:
    """Single-asset simulator tracking daily value and drawdown."""

    initial_capital: float = 100000.0
    position_fraction: float = 0.2
    hedge_ratio: float = 0.0
    cash: float = field(init=False)
    position_value: float = field(init=False)
    peak_value: float = field(init=False)
    value_history: List[float] = field(default_factory=list)
    return_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cash = float(self.initial_capital * (1.0 - self.position_fraction))
        self.position_value = float(self.initial_capital * self.position_fraction)
        total = self.cash + self.position_value
        self.peak_value = total
        self.value_history = [total]
        self.return_history = [0.0]

    @property
    def total_value(self) -> float:
        return float(self.cash + self.position_value)

    @property
    def leverage_ratio(self) -> float:
        return float(self.position_value / max(self.total_value, 1e-9))

    @property
    def drawdown(self) -> float:
        return float(max(0.0, 1.0 - (self.total_value / max(self.peak_value, 1e-9))))

    def set_risk_profile(self, max_position_size: float, hedge_ratio: float) -> None:
        capped_fraction = float(np.clip(max_position_size, 0.0, 1.0))
        target_position = self.total_value * capped_fraction
        delta = target_position - self.position_value
        self.position_value += delta
        self.cash -= delta
        self.hedge_ratio = float(np.clip(hedge_ratio, 0.0, 1.0))

    def step_day(self, asset_return: float, stop_loss: float) -> Dict[str, float]:
        prev_total = self.total_value
        hedge_cost = self.hedge_ratio * 0.0005
        effective_return = float(asset_return * (1.0 - self.hedge_ratio) - hedge_cost)
        self.position_value *= 1.0 + effective_return

        if effective_return < -abs(stop_loss):
            self.cash += self.position_value
            self.position_value = 0.0

        total = self.total_value
        self.peak_value = max(self.peak_value, total)
        daily_return = (total / max(prev_total, 1e-9)) - 1.0
        self.value_history.append(total)
        self.return_history.append(float(daily_return))

        return {
            "portfolio_value": total,
            "daily_return": float(daily_return),
            "drawdown": self.drawdown,
            "leverage": self.leverage_ratio,
        }
