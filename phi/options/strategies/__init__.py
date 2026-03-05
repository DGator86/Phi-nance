"""Options strategy library."""

from .base import Leg, OptionStrategy
from .advanced import ButterflySpread, CalendarSpread, Collar, CoveredCall, DiagonalSpread, ProtectivePut
from .combos import IronCondor, Straddle, Strangle
from .single import SingleLeg
from .spreads import VerticalSpread

__all__ = [
    "Leg",
    "OptionStrategy",
    "SingleLeg",
    "VerticalSpread",
    "Straddle",
    "Strangle",
    "IronCondor",
    "ButterflySpread",
    "CalendarSpread",
    "DiagonalSpread",
    "CoveredCall",
    "ProtectivePut",
    "Collar",
]
