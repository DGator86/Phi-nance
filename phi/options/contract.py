from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class OptionType(str, Enum):
    """Supported vanilla option types."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionContract:
    """European option contract specification."""

    underlying: str
    option_type: OptionType
    strike: float
    expiry: date
    style: str = "european"
    multiplier: int = 100

    def time_to_expiry(self, as_of: date) -> float:
        """Return year fraction to expiry using ACT/365 day count."""
        return max((self.expiry - as_of).days / 365.0, 0.0)
