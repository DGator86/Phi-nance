"""Spread strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Leg, OptionStrategy


@dataclass
class VerticalSpread(OptionStrategy):
    option_type: str
    lower_strike: float
    upper_strike: float
    expiry: float
    quantity: int = 1
    bullish: bool = True

    def legs(self):
        if self.bullish:
            return [
                Leg(self.option_type, "buy", self.lower_strike, self.expiry, self.quantity),
                Leg(self.option_type, "sell", self.upper_strike, self.expiry, self.quantity),
            ]
        return [
            Leg(self.option_type, "sell", self.lower_strike, self.expiry, self.quantity),
            Leg(self.option_type, "buy", self.upper_strike, self.expiry, self.quantity),
        ]
