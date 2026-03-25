"""Hard veto rules before scoring (extend per deployment)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from phi.force_field.schemas import MotionState, RegimeTensor


@dataclass
class VetoThresholds:
    iron_condor_max_live_shock: float = 0.35
    iron_condor_max_neg_gamma: float = 0.45
    iron_condor_max_thin_liq: float = 0.45
    straddle_max_pin: float = 0.85
    straddle_min_iv_rank: float = 0.0
    calendar_min_term_edge: float = -0.5
    calendar_max_thin_liq: float = 0.55
    calendar_max_shock: float = 0.4


@dataclass
class VetoEngine:
    thresholds: VetoThresholds = field(default_factory=VetoThresholds)

    def check(
        self,
        strategy_name: str,
        regime: RegimeTensor,
        motion: MotionState,
        features: Mapping[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        news = regime.news
        gr = regime.greek
        liq = regime.liquidity

        if strategy_name in {"iron_condor", "iron_butterfly", "call_credit_spread", "put_credit_spread"}:
            if float(news.get("live_shock", 0.0)) > self.thresholds.iron_condor_max_live_shock:
                reasons.append("credit_structure_veto_live_shock")
            if float(gr.get("neg_gamma_accel", 0.0)) > self.thresholds.iron_condor_max_neg_gamma:
                reasons.append("credit_structure_veto_neg_gamma")
            if float(liq.get("thin_liquidity", 0.0)) > self.thresholds.iron_condor_max_thin_liq:
                reasons.append("credit_structure_veto_thin_liquidity")

        if strategy_name in {"long_straddle", "long_strangle"}:
            iv_rank = float(features.get("iv_rank", 0.5))
            if motion.pinning_strength > self.thresholds.straddle_max_pin:
                reasons.append("long_vol_veto_high_pin")
            if iv_rank < self.thresholds.straddle_min_iv_rank:
                reasons.append("long_vol_veto_iv_rank_floor")

        if strategy_name == "calendar":
            term_edge = float(features.get("term_structure_edge", 0.0))
            if term_edge < self.thresholds.calendar_min_term_edge:
                reasons.append("calendar_veto_no_term_edge")
            if float(liq.get("thin_liquidity", 0.0)) > self.thresholds.calendar_max_thin_liq:
                reasons.append("calendar_veto_thin_liquidity")
            if float(news.get("live_shock", 0.0)) > self.thresholds.calendar_max_shock:
                reasons.append("calendar_veto_shock")

        return reasons
