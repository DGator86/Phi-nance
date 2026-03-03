"""
Composer V1 — direction distribution, drift (bps), vol cones (50/75/90), annualized σ.

Cones never inverted; packet fully populated for all horizons (confidence may be low).
"""

from __future__ import annotations

from typing import Any

from phinence.contracts.projection_packet import (
    Horizon,
    HorizonProjection,
    ProjectionPacket,
    SchemaVersion,
    VolCone,
    make_empty_horizon,
)
from phinence.mfm.merger import MarketFieldMap

from phinence.composer.calibration import direction_from_mfm, drift_bps_from_mfm


def _ensure_cone_not_inverted(cone: VolCone) -> VolCone:
    """Enforce p50 <= p75 <= p90."""
    p50, p75, p90 = cone.p50_bps, cone.p75_bps, cone.p90_bps
    if p50 > p75:
        p75 = p50
    if p75 > p90:
        p90 = p75
    return VolCone(p50_bps=p50, p75_bps=p75, p90_bps=p90, annualized_sigma=cone.annualized_sigma)


class Composer:
    """MFM → ProjectionPacket (per-horizon direction, drift, cones)."""

    def __init__(self, default_annual_sigma: float = 0.20) -> None:
        self.default_annual_sigma = default_annual_sigma

    def run(
        self,
        mfm: MarketFieldMap,
        horizons: list[Horizon] | None = None,
    ) -> ProjectionPacket:
        """
        Produce ProjectionPacket. Uses calibration (MFM → direction, drift); cones from sigma.
        """
        horizons_list = horizons or [Horizon.INTRADAY_1M, Horizon.INTRADAY_5M, Horizon.DAILY]
        out: list[HorizonProjection] = []
        for h in horizons_list:
            hp = make_empty_horizon(h)
            hp.direction = direction_from_mfm(mfm, h)
            hp.drift_bps = drift_bps_from_mfm(mfm, h)
            sigma = self.default_annual_sigma
            if h == Horizon.INTRADAY_1M:
                scale = 1 / (252 * 390) ** 0.5
            elif h == Horizon.INTRADAY_5M:
                scale = (5 / 390) ** 0.5 / 252 ** 0.5
            else:
                scale = 1 / 252 ** 0.5
            hp.cone = _ensure_cone_not_inverted(VolCone(
                p50_bps=sigma * scale * 67,
                p75_bps=sigma * scale * 115,
                p90_bps=sigma * scale * 164,
                annualized_sigma=sigma,
            ))
            hp.confidence = 0.5 if (mfm.regime or mfm.liquidity) else 0.0
            out.append(hp)
        return ProjectionPacket(
            schema_version=SchemaVersion.V1,
            ticker=mfm.ticker,
            as_of=mfm.as_of,
            horizons=out,
            meta={},
        )
