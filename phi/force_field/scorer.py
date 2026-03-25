"""Score strategies from motion + regime + auxiliary features."""

from __future__ import annotations

from typing import Mapping

from phi.force_field.schemas import MotionState, RegimeTensor
from phi.force_field.strategy_profiles import STRATEGY_PROFILES, StrategyProfile


def score_strategy(
    profile: StrategyProfile,
    motion: MotionState,
    regime: RegimeTensor,
    features: Mapping[str, float],
    execution_penalty: float = 0.0,
    event_mismatch: float = 0.0,
) -> float:
    term_edge = float(features.get("term_structure_edge", 0.0))
    p: StrategyProfile = profile
    score = 0.0
    score += p["directional_need"] * motion.expected_direction
    score += p["expansion_need"] * motion.expected_expansion
    range_proxy = 1.0 - min(1.0, abs(motion.expected_direction))
    score += p["range_preference"] * range_proxy
    score += p["pin_preference"] * motion.pinning_strength
    score += p["term_structure_need"] * term_edge
    score += p["liquidity_need"] * max(0.0, 1.0 - motion.execution_difficulty)
    score -= 0.55 * float(regime.news.get("live_shock", 0.0))
    score -= execution_penalty
    score -= event_mismatch
    return score


def score_all_strategies(
    motion: MotionState,
    regime: RegimeTensor,
    features: Mapping[str, float],
    *,
    execution_penalty: float = 0.0,
    profiles: dict[str, StrategyProfile] | None = None,
) -> dict[str, float]:
    book = profiles or STRATEGY_PROFILES
    return {
        name: score_strategy(prof, motion, regime, features, execution_penalty=execution_penalty)
        for name, prof in book.items()
    }
