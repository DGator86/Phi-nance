"""
ProjectionPacket — hard boundary of the projection system.

No strategy selection, no routing, no sizing. All required fields present;
confidences may be 0.0 when data is missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SchemaVersion(str, Enum):
    """Packet schema version for replay and compatibility."""
    V1 = "1.0.0"


class Horizon(str, Enum):
    """Projection horizons; WF windows are horizon-specific."""
    INTRADAY_1M = "1m"
    INTRADAY_5M = "5m"
    DAILY = "1d"


class DirectionProbs(BaseModel):
    """Direction distribution (up/down/flat)."""
    up: float = Field(ge=0.0, le=1.0, description="P(up)")
    down: float = Field(ge=0.0, le=1.0, description="P(down)")
    flat: float = Field(ge=0.0, le=1.0, description="P(flat)")


class VolCone(BaseModel):
    """Volatility cone at a given confidence level."""
    p50_bps: float = Field(ge=0.0, description="50% cone width in bps")
    p75_bps: float = Field(ge=0.0, description="75% cone width in bps")
    p90_bps: float = Field(ge=0.0, description="90% cone width in bps")
    annualized_sigma: float = Field(ge=0.0, description="Annualized vol")


class HorizonProjection(BaseModel):
    """Per-horizon outputs: direction, drift, cones."""
    horizon: Horizon
    direction: DirectionProbs
    drift_bps: float
    cone: VolCone
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ProjectionPacket(BaseModel):
    """
    Canonical output of the projection pipeline. Generate valid packets
    with all required fields; use confidence=0.0 when data is missing.
    """

    schema_version: SchemaVersion = Field(default=SchemaVersion.V1, alias="schema_version")
    ticker: str = Field(min_length=1)
    as_of: datetime = Field(description="Timestamp of projection")
    horizons: list[HorizonProjection] = Field(min_length=1)
    # Optional engine-level context (for debugging/ablations); not used for trading decisions
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def get_horizon(self, h: Horizon) -> HorizonProjection | None:
        """Return projection for a given horizon."""
        for hp in self.horizons:
            if hp.horizon == h:
                return hp
        return None


def make_empty_horizon(horizon: Horizon) -> HorizonProjection:
    """Build a valid horizon projection with zero confidence when data missing."""
    return HorizonProjection(
        horizon=horizon,
        direction=DirectionProbs(up=1.0 / 3, down=1.0 / 3, flat=1.0 / 3),
        drift_bps=0.0,
        cone=VolCone(p50_bps=0.0, p75_bps=0.0, p90_bps=0.0, annualized_sigma=0.0),
        confidence=0.0,
    )


def make_stub_packet(ticker: str, as_of: datetime | None = None) -> ProjectionPacket:
    """Generate a valid ProjectionPacket with all required fields; confidences 0.0."""
    as_of = as_of or datetime.now(timezone.utc)
    return ProjectionPacket(
        schema_version=SchemaVersion.V1,
        ticker=ticker,
        as_of=as_of,
        horizons=[
            make_empty_horizon(Horizon.INTRADAY_1M),
            make_empty_horizon(Horizon.INTRADAY_5M),
            make_empty_horizon(Horizon.DAILY),
        ],
        meta={},
    )
