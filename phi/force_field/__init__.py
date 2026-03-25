"""Multi-regime tensor → force field → mass → motion → defined-risk strategy ranking."""

from __future__ import annotations

from phi.force_field.acceleration import derive_motion
from phi.force_field.force_mapper import ForceMapper, default_force_weights
from phi.force_field.mass_model import MassModel
from phi.force_field.pipeline import rank_strategies, run_engine, run_pipeline
from phi.force_field.potential_field import GaussianWell, neg_gradient_1d, total_potential
from phi.force_field.regime_engines import (
    build_regime_tensor_from_features,
    greek_from_features,
    indicator_from_features,
    industry_from_features,
    liquidity_from_features,
    neutral_regime_tensor,
    news_from_features,
)
from phi.force_field.schemas import (
    EngineState,
    ForceVector,
    MassState,
    MotionState,
    RegimeTensor,
    StrategyScoreResult,
)
from phi.force_field.scorer import score_all_strategies, score_strategy
from phi.force_field.strategy_aliases import (
    FORCE_FIELD_TO_LEGACY_NAME,
    LEGACY_NAME_TO_FORCE_FIELD,
    to_legacy_strategy_name,
)
from phi.force_field.strategy_profiles import STRATEGY_PROFILES, list_strategies
from phi.force_field.taxonomy import (
    FORCE_DIMS,
    GREEK_SUBREGIMES,
    INDICATOR_SUBREGIMES,
    INDUSTRY_SUBREGIMES,
    LIQUIDITY_SUBREGIMES,
    NEWS_SUBREGIMES,
    ensure_simplex,
    softmax,
    uniform_simplex,
    validate_simplex,
)
from phi.force_field.vetoes import VetoEngine, VetoThresholds

__all__ = [
    "EngineState",
    "FORCE_DIMS",
    "FORCE_FIELD_TO_LEGACY_NAME",
    "ForceMapper",
    "ForceVector",
    "GaussianWell",
    "GREEK_SUBREGIMES",
    "INDICATOR_SUBREGIMES",
    "INDUSTRY_SUBREGIMES",
    "LEGACY_NAME_TO_FORCE_FIELD",
    "LIQUIDITY_SUBREGIMES",
    "MassModel",
    "MassState",
    "MotionState",
    "NEWS_SUBREGIMES",
    "RegimeTensor",
    "STRATEGY_PROFILES",
    "StrategyScoreResult",
    "VetoEngine",
    "VetoThresholds",
    "build_regime_tensor_from_features",
    "default_force_weights",
    "derive_motion",
    "ensure_simplex",
    "greek_from_features",
    "indicator_from_features",
    "industry_from_features",
    "liquidity_from_features",
    "list_strategies",
    "neutral_regime_tensor",
    "news_from_features",
    "neg_gradient_1d",
    "rank_strategies",
    "run_engine",
    "run_pipeline",
    "score_all_strategies",
    "score_strategy",
    "softmax",
    "to_legacy_strategy_name",
    "total_potential",
    "uniform_simplex",
    "validate_simplex",
]
