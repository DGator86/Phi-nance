"""Option position representation and valuation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from phi.logging import get_logger

from .contract import OptionContract, OptionType
from .pricing import black_scholes_price

logger = get_logger(__name__)


@dataclass
class OptionPosition:
    """Represents an open options position."""

    symbol: str = ""
    option_type: str = "CALL"  # 'CALL' or 'PUT'
    strike: float = 0.0
    expiration: date = field(default_factory=date.today)
    quantity: int = 0  # positive for long, negative for short
    entry_price: float = 0.0
    entry_date: date = field(default_factory=date.today)
    exit_price: Optional[float] = None
    exit_date: Optional[date] = None
    entry_delta: Optional[float] = None
    entry_gamma: Optional[float] = None
    entry_theta: Optional[float] = None
    entry_vega: Optional[float] = None
    entry_rho: Optional[float] = None
    multiplier: int = 100

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def contract(self) -> OptionContract:
        return OptionContract(
            underlying=self.symbol,
            option_type=OptionType.CALL if self.option_type.upper() == "CALL" else OptionType.PUT,
            strike=self.strike,
            expiry=self.expiration,
            multiplier=self.multiplier,
        )

    @property
    def entry_cost(self) -> float:
        return self.entry_price * self.quantity * self.multiplier

    def is_expired(self, current_date: date) -> bool:
        return current_date > self.expiration

    def mark_to_market(
        self,
        current_price: Optional[float] = None,
        as_of: Optional[date] = None,
        r: float = 0.02,
        sigma: float = 0.3,
        underlying_price: Optional[float] = None,
    ) -> float:
        """Return current market value of the position."""
        if current_price is not None:
            return self.quantity * current_price * self.multiplier

        if as_of is None or underlying_price is None:
            raise ValueError("as_of and underlying_price are required when current_price is not provided")

        unit_price = black_scholes_price(
            S=underlying_price,
            K=self.strike,
            T=max((self.expiration - as_of).days / 365.0, 0.0),
            r=r,
            sigma=sigma,
            option_type=self.contract.option_type,
        )
        return unit_price * self.quantity * self.multiplier

    def close(self, exit_price: float, exit_date: date) -> None:
        self.exit_price = exit_price
        self.exit_date = exit_date

    @classmethod
    def from_contract(cls, contract: OptionContract, quantity: int, entry_cost: float, entry_date: Optional[date] = None) -> "OptionPosition":
        contracts = abs(quantity) if quantity else 0
        unit_price = entry_cost / (contracts * contract.multiplier) if contracts else 0.0
        return cls(
            symbol=contract.underlying,
            option_type=contract.option_type.value.upper(),
            strike=contract.strike,
            expiration=contract.expiry,
            quantity=quantity,
            entry_price=unit_price,
            entry_date=entry_date or date.today(),
            multiplier=contract.multiplier,
        )
