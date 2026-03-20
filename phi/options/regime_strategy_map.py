"""Backward-compatible import surface for regime-based options strategy mapping."""

from phi.regime.strategy_mapping import (
    APPROVED_STRATEGIES,
    REGIME_STRATEGY_MAP,
    is_approved_strategy,
    map_regime_probabilities_to_strategies,
    strategies_for_regime,
)

__all__ = [
    "APPROVED_STRATEGIES",
    "REGIME_STRATEGY_MAP",
    "strategies_for_regime",
    "map_regime_probabilities_to_strategies",
    "is_approved_strategy",
]
