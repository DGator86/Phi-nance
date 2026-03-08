"""Options strategy library."""

from phi.logging import get_logger

logger = get_logger(__name__)

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
