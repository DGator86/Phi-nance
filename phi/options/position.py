"""Option position representation and valuation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .contract import OptionContract
from .pricing import black_scholes_price


@dataclass
class OptionPosition:
    contract: OptionContract
    quantity: int
    entry_cost: float

    def mark_to_market(self, underlying_price: float, as_of: date, r: float, sigma: float) -> float:
        """Return marked value in quote currency (signed by quantity)."""
        unit_price = black_scholes_price(
            S=underlying_price,
            K=self.contract.strike,
            T=self.contract.time_to_expiry(as_of),
            r=r,
            sigma=sigma,
            option_type=self.contract.option_type,
        )
        return unit_price * self.quantity * self.contract.multiplier

    @staticmethod
    def pnl(current_value: float, cost_basis: float) -> float:
        return current_value - cost_basis
