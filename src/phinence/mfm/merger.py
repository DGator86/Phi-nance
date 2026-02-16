"""
MarketFieldMap — merger of engine outputs: namespaced fields, landmarks, steering vectors.

Same inputs → same MFM (deterministic, replayable).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MarketFieldMap(BaseModel):
    """Single object the composer consumes. No strategy/routing/sizing."""

    ticker: str = Field(min_length=1)
    as_of: datetime
    liquidity: dict[str, Any] = Field(default_factory=dict)
    regime: dict[str, Any] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    hedge: dict[str, Any] = Field(default_factory=dict)
    landmarks: dict[str, Any] = Field(default_factory=dict)
    steering: dict[str, Any] = Field(default_factory=dict)

    def get_namespaced(self, namespace: str) -> dict[str, Any]:
        """Engine namespace: liquidity, regime, sentiment, hedge."""
        return getattr(self, namespace, {}) or {}


def build_mfm(
    ticker: str,
    as_of: datetime,
    liquidity: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
    sentiment: dict[str, Any] | None = None,
    hedge: dict[str, Any] | None = None,
) -> MarketFieldMap:
    """Build MFM from engine outputs. Deterministic merge; same inputs → same MFM."""
    liq = liquidity or {}
    hed = hedge or {}
    swings = liq.get("swings") or {}
    return MarketFieldMap(
        ticker=ticker,
        as_of=as_of,
        liquidity=liq,
        regime=regime or {},
        sentiment=sentiment or {},
        hedge=hed,
        landmarks={
            **swings,
            "poc": liq.get("poc"),
            "vah": liq.get("vah"),
            "val": liq.get("val"),
            "epp_proxy": hed.get("epp_proxy"),
        },
        steering={},
    )
