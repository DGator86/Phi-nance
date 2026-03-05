"""Single-leg options strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Leg, OptionStrategy


@dataclass
class SingleLeg(OptionStrategy):
    option_type: str
    action: str
    strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [Leg(self.option_type, self.action, self.strike, self.expiry, self.quantity)]
