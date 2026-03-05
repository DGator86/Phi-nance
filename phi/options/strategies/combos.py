"""Combination options strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Leg, OptionStrategy


@dataclass
class Straddle(OptionStrategy):
    strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg("call", "buy", self.strike, self.expiry, self.quantity),
            Leg("put", "buy", self.strike, self.expiry, self.quantity),
        ]


@dataclass
class Strangle(OptionStrategy):
    lower_strike: float
    upper_strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg("put", "buy", self.lower_strike, self.expiry, self.quantity),
            Leg("call", "buy", self.upper_strike, self.expiry, self.quantity),
        ]


@dataclass
class IronCondor(OptionStrategy):
    put_wing: float
    put_short: float
    call_short: float
    call_wing: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg("put", "buy", self.put_wing, self.expiry, self.quantity),
            Leg("put", "sell", self.put_short, self.expiry, self.quantity),
            Leg("call", "sell", self.call_short, self.expiry, self.quantity),
            Leg("call", "buy", self.call_wing, self.expiry, self.quantity),
        ]
