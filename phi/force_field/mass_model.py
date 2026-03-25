"""Effective mass from size, liquidity, gamma stabilization, event risk."""

from __future__ import annotations

import math
from typing import Mapping

from phi.force_field.schemas import MassState, RegimeTensor


class MassModel:
    def __init__(
        self,
        w_cap: float = 0.35,
        w_adv: float = 0.30,
        w_depth: float = 0.20,
        w_gamma: float = 0.25,
        w_event: float = 0.30,
        base_mass: float = 0.15,
    ) -> None:
        self.w_cap = w_cap
        self.w_adv = w_adv
        self.w_depth = w_depth
        self.w_gamma = w_gamma
        self.w_event = w_event
        self.base_mass = base_mass

    def compute(self, features: Mapping[str, float], regime: RegimeTensor) -> MassState:
        cap_bn = max(1.0, _f(features, "market_cap_billions", 10.0))
        adv_m = max(1e6, _f(features, "adv_dollar", 50e6))
        depth = _f(features, "depth_score", 0.5)
        event_sens = (
            float(regime.news.get("live_shock", 0.0))
            + float(regime.news.get("scheduled_event", 0.0))
            + 0.5 * float(regime.news.get("macro_risk", 0.0))
        )
        gamma_stab = float(regime.greek.get("pos_gamma_pin", 0.0)) + 0.5 * float(
            regime.greek.get("vol_compression", 0.0)
        )
        liq_score = float(regime.liquidity.get("thick_liquidity", 0.0)) + 0.35 * (
            1.0 - float(regime.liquidity.get("thin_liquidity", 0.0))
        )

        cap_part = self.w_cap * math.log10(cap_bn)
        adv_part = self.w_adv * (math.log10(adv_m) - 6.0) / 3.0
        depth_part = self.w_depth * max(0.0, min(1.0, depth))
        gamma_part = self.w_gamma * max(0.0, min(1.0, gamma_stab))
        evt_penalty = self.w_event * max(0.0, min(1.5, event_sens))

        raw = self.base_mass + cap_part + adv_part + depth_part + gamma_part - evt_penalty
        raw = max(0.08, raw)
        normalized = max(0.05, min(2.0, raw)) / 2.0
        return MassState(
            raw_mass=raw,
            normalized_mass=normalized,
            cap_score=cap_part,
            liquidity_score=liq_score,
            gamma_stability_score=gamma_part,
            event_sensitivity_score=evt_penalty,
        )


def _f(features: Mapping[str, float], key: str, default: float) -> float:
    return float(features.get(key, default))
