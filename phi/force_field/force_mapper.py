"""Map flattened regime probabilities to a low-dimensional force vector."""

from __future__ import annotations

from typing import Mapping

from phi.force_field.schemas import ForceVector, RegimeTensor
from phi.force_field.taxonomy import FORCE_DIMS


class ForceMapper:
    """Linear map: each force dimension is a dot product against flat regime keys."""

    def __init__(self, weights: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self.weights: dict[str, dict[str, float]] = (
            {k: dict(v) for k, v in weights.items()} if weights else default_force_weights()
        )

    def map(self, regime: RegimeTensor) -> ForceVector:
        flat = regime.flattened()
        out: dict[str, float] = {d: 0.0 for d in FORCE_DIMS}
        for dim in FORCE_DIMS:
            row = self.weights.get(dim, {})
            out[dim] = sum(float(flat.get(k, 0.0)) * w for k, w in row.items())
        return ForceVector(
            directional=out["directional"],
            magnet=out["magnet"],
            expansion=out["expansion"],
            friction=out["friction"],
            systemic=out["systemic"],
        )


def default_force_weights() -> dict[str, dict[str, float]]:
    """Interpretable rule-based prior (tune / replace via calibration)."""
    return {
        "directional": {
            "ind.trend_up": 1.2,
            "ind.trend_down": -1.2,
            "ind.breakout_setup": 0.6,
            "ind.reversal_setup": -0.2,
            "gr.neg_gamma_accel": 0.35,
            "gr.vol_expansion": 0.15,
        },
        "magnet": {
            "gr.pos_gamma_pin": 1.0,
            "liq.pool_magnet": 0.8,
            "ind.range": 0.25,
            "gr.vol_compression": 0.15,
        },
        "expansion": {
            "gr.vol_expansion": 1.0,
            "gr.neg_gamma_accel": 0.45,
            "news.live_shock": 0.55,
            "news.scheduled_event": 0.35,
            "ind.compression": 0.4,
            "gr.pos_gamma_pin": -0.55,
            "gr.vol_compression": -0.35,
            "ind.range": -0.2,
        },
        "friction": {
            "liq.thin_liquidity": 1.0,
            "liq.sweep_risk": 0.8,
            "liq.vacuum_zone": 0.6,
            "news.live_shock": 0.4,
            "news.macro_risk": 0.25,
            "liq.thick_liquidity": -0.55,
            "liq.chop_friction": 0.15,
        },
        "systemic": {
            "indus.strong_index_lock": 0.9,
            "indus.strong_sector_lock": 0.65,
            "indus.sector_rotation": 0.35,
            "indus.local_idiosyncratic": -0.15,
        },
    }
