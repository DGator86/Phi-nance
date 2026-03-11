"""Options strategy selection helpers for detailed market regimes."""

from __future__ import annotations

from collections.abc import Iterable

from phi.regime.regime_definitions import BASE_REGIMES, VOLATILITY_REGIMES

APPROVED_STRATEGIES: tuple[str, ...] = (
    "Long Call",
    "Bull Call Spread",
    "Bull Put Spread",
    "Long Put",
    "Bear Put Spread",
    "Bear Call Spread",
    "Long Straddle",
    "Long Strangle",
    "Iron Condor",
    "Iron Butterfly",
    "Long Call Butterfly",
    "Long Put Butterfly",
    "Calendar Spread",
    "Diagonal Spread",
    "Cash-Secured Put",
)

REGIME_STRATEGY_MAP: dict[str, tuple[str, ...]] = {
    "BULL_LOW_VOL": ("Bull Put Spread", "Cash-Secured Put", "Long Call", "Bull Call Spread"),
    "BULL_NORMAL_VOL": ("Bull Put Spread", "Long Call", "Bull Call Spread", "Diagonal Spread"),
    "BULL_HIGH_VOL": ("Long Call", "Bull Call Spread", "Diagonal Spread"),
    "BEAR_LOW_VOL": ("Bear Call Spread", "Long Put", "Bear Put Spread"),
    "BEAR_NORMAL_VOL": ("Bear Call Spread", "Long Put", "Bear Put Spread", "Diagonal Spread"),
    "BEAR_HIGH_VOL": ("Long Put", "Bear Put Spread"),
    "RANGING_LOW_VOL": (
        "Iron Condor",
        "Iron Butterfly",
        "Long Call Butterfly",
        "Long Put Butterfly",
        "Calendar Spread",
    ),
    "RANGING_NORMAL_VOL": ("Iron Condor", "Calendar Spread", "Diagonal Spread"),
    "RANGING_HIGH_VOL": ("Long Straddle", "Long Strangle"),
}


def _validate_detailed_regime(regime_label: str) -> str:
    label = regime_label.strip().upper()
    if label in REGIME_STRATEGY_MAP:
        return label

    chunks = label.split("_")
    if len(chunks) >= 2:
        trend = chunks[0]
        vol = "_".join(chunks[1:])
        if trend in BASE_REGIMES and vol in VOLATILITY_REGIMES:
            return label

    raise ValueError(f"Unknown detailed regime '{regime_label}'")


def strategies_for_regime(regime_label: str) -> list[str]:
    """Return prioritized approved strategies for a detailed regime label."""
    label = _validate_detailed_regime(regime_label)
    return list(REGIME_STRATEGY_MAP.get(label, ()))


def map_regime_probabilities_to_strategies(regime_probs: dict[str, float]) -> list[tuple[str, float]]:
    """Aggregate strategy scores from detailed regime probabilities."""
    scores: dict[str, float] = {}
    for regime_label, prob in regime_probs.items():
        label = _validate_detailed_regime(regime_label)
        for rank, strategy in enumerate(REGIME_STRATEGY_MAP.get(label, ())):
            weight = float(prob) * (1.0 / (rank + 1))
            scores[strategy] = scores.get(strategy, 0.0) + weight
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def is_approved_strategy(strategy_name: str, approved: Iterable[str] = APPROVED_STRATEGIES) -> bool:
    """Check whether a strategy is in the curated defined-risk list."""
    return strategy_name in set(approved)
