"""Base classes for options strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from phi.options.models.black_scholes import greeks, price_european


@dataclass
class Leg:
    option_type: str
    action: str
    strike: float
    expiry: float
    quantity: int = 1


class OptionStrategy(ABC):
    @abstractmethod
    def legs(self) -> List[Leg]:
        ...

    def validate(self) -> bool:
        lgs = self.legs()
        return bool(lgs) and all(l.option_type in {"call", "put"} and l.action in {"buy", "sell"} for l in lgs)

    def net_premium(self, S: float, r: float, iv_surface) -> float:
        total = 0.0
        for leg in self.legs():
            sigma = iv_surface.get_iv(leg.strike, leg.expiry)
            p = price_european(leg.option_type, S, leg.strike, leg.expiry, r, sigma)
            sign = 1.0 if leg.action == "buy" else -1.0
            total += sign * p * leg.quantity * 100.0
        return total

    def greeks(self, S: float, r: float, iv_surface) -> dict:
        agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for leg in self.legs():
            sigma = iv_surface.get_iv(leg.strike, leg.expiry)
            g = greeks(leg.option_type, S, leg.strike, leg.expiry, r, sigma)
            sign = 1.0 if leg.action == "buy" else -1.0
            for k in agg:
                agg[k] += sign * g[k] * leg.quantity * 100.0
        return agg
