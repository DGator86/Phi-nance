"""Advanced multi-leg options strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Leg, OptionStrategy


@dataclass
class ButterflySpread(OptionStrategy):
    option_type: str
    lower_strike: float
    middle_strike: float
    upper_strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg(self.option_type, "buy", self.lower_strike, self.expiry, self.quantity),
            Leg(self.option_type, "sell", self.middle_strike, self.expiry, self.quantity * 2),
            Leg(self.option_type, "buy", self.upper_strike, self.expiry, self.quantity),
        ]


@dataclass
class CalendarSpread(OptionStrategy):
    option_type: str
    strike: float
    near_expiry: float
    far_expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg(self.option_type, "sell", self.strike, self.near_expiry, self.quantity),
            Leg(self.option_type, "buy", self.strike, self.far_expiry, self.quantity),
        ]


@dataclass
class DiagonalSpread(OptionStrategy):
    option_type: str
    short_strike: float
    long_strike: float
    near_expiry: float
    far_expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg(self.option_type, "sell", self.short_strike, self.near_expiry, self.quantity),
            Leg(self.option_type, "buy", self.long_strike, self.far_expiry, self.quantity),
        ]


@dataclass
class CoveredCall(OptionStrategy):
    strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [Leg("call", "sell", self.strike, self.expiry, self.quantity)]


@dataclass
class ProtectivePut(OptionStrategy):
    strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [Leg("put", "buy", self.strike, self.expiry, self.quantity)]


@dataclass
class Collar(OptionStrategy):
    put_strike: float
    call_strike: float
    expiry: float
    quantity: int = 1

    def legs(self):
        return [
            Leg("put", "buy", self.put_strike, self.expiry, self.quantity),
            Leg("call", "sell", self.call_strike, self.expiry, self.quantity),
        ]
