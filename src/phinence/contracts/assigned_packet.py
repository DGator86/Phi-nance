"""
AssignedPacket — typed bundle from AssignmentEngine to each engine.

Coverage flags and warnings; missing data never crashes—coverage drops, confidence → 0.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CoverageFlag(str, Enum):
    """Per-source coverage status."""
    FULL = "full"
    PARTIAL = "partial"
    MISSING = "missing"
    STALE = "stale"


class AssignedPacket(BaseModel):
    """
    Output of AssignmentEngine: what each engine receives. Includes
    1m bars, derived 5m bars (resampled), coverage flags, warnings.
    """

    ticker: str = Field(min_length=1)
    as_of: datetime
    bars_1m: list[dict[str, Any]] = Field(default_factory=list)
    bars_5m: list[dict[str, Any]] = Field(default_factory=list)
    coverage_1m: CoverageFlag = Field(default=CoverageFlag.MISSING)
    coverage_5m: CoverageFlag = Field(default=CoverageFlag.MISSING)
    chain_snapshot: dict[str, Any] | None = Field(default=None)
    chain_coverage: CoverageFlag = Field(default=CoverageFlag.MISSING)
    warnings: list[str] = Field(default_factory=list)

    def has_sufficient_bars_1m(self, min_bars: int = 5) -> bool:
        """No-gap sanity: need at least min_bars for RTH sequence."""
        return self.coverage_1m == CoverageFlag.FULL and len(self.bars_1m) >= min_bars

    def has_sufficient_bars_5m(self, min_bars: int = 5) -> bool:
        return self.coverage_5m == CoverageFlag.FULL and len(self.bars_5m) >= min_bars
