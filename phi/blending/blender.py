"""Signal blending utilities."""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)

ALLOWED_METHODS = {"weighted_sum", "voting", "regime_weighted"}

# Backward-compatible default; callers should pass explicit regime_boosts.
DEFAULT_REGIME_BOOSTS: Dict[str, Dict[str, float]] = {
    "TREND_UP": {"MACD": 1.2, "Dual SMA": 1.2},
    "TREND_DN": {"MACD": 1.2, "Dual SMA": 1.2},
    "RANGE": {"RSI": 1.2, "Bollinger": 1.2, "Mean Reversion": 1.2, "VWAP": 1.1},
    "BREAKOUT_UP": {"Breakout": 1.2, "MACD": 1.1},
    "BREAKOUT_DN": {"Breakout": 1.2, "MACD": 1.1},
}


def _validate_signals(signals: pd.DataFrame) -> None:
    if not isinstance(signals, pd.DataFrame):
        raise ValueError("signals must be a pandas DataFrame")
    if signals.empty or signals.columns.empty:
        raise ValueError("signals must be a non-empty DataFrame with at least one column")

    numeric = signals.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy()).all():
        logger.warning("signals contains non-finite values (NaN/Inf); blending may fail")

    all_nan_columns = [col for col in signals.columns if signals[col].isna().all()]
    if all_nan_columns:
        raise ValueError(f"signals contains all-NaN columns: {all_nan_columns}")


def _validate_weights(columns: pd.Index, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if weights is None:
        raise ValueError("weights are required for this blend method")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a non-empty dict")

    col_set = set(columns)
    weight_set = set(weights.keys())
    if col_set != weight_set:
        missing = sorted(col_set - weight_set)
        extra = sorted(weight_set - col_set)
        raise ValueError(f"weights keys must exactly match signal columns; missing={missing}, extra={extra}")

    parsed_weights = {k: float(v) for k, v in weights.items()}
    total = sum(parsed_weights.values())
    if np.isclose(total, 0.0):
        raise ValueError("weights sum cannot be zero")

    if abs(total - 1.0) > 1e-8:
        logger.warning("weights sum is %.6f (expected 1.0); normalizing", total)
        parsed_weights = {k: v / total for k, v in parsed_weights.items()}

    return parsed_weights


def _weighted_sum(signals: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    logger.debug("Running weighted_sum blend for %d signals", len(signals.columns))
    weight_series = pd.Series(weights).reindex(signals.columns)
    return (signals * weight_series).sum(axis=1)


def _resolve_regime(
    regime: Optional[str],
    regime_probs: Optional[pd.DataFrame],
) -> Optional[str]:
    if regime is not None:
        return regime
    if regime_probs is None or regime_probs.empty:
        return None
    aligned = regime_probs.dropna(how="all")
    if aligned.empty:
        return None
    inferred = str(aligned.iloc[-1].idxmax())
    logger.debug("Inferred regime from regime_probs: %s", inferred)
    return inferred


def blend_signals(
    signals: pd.DataFrame,
    method: Literal["weighted_sum", "voting", "regime_weighted"] = "weighted_sum",
    weights: Optional[Dict[str, float]] = None,
    regime: Optional[str] = None,
    regime_boosts: Optional[Dict[str, Dict[str, float]]] = None,
    regime_probs: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Blend multiple indicator signals into a composite signal.

    Supports legacy positional usage ``blend_signals(signals, weights, method)``.
    """
    # Legacy positional compatibility: blend_signals(signals, weights, method)
    if isinstance(method, dict):
        legacy_weights = method
        if isinstance(weights, str):
            method = weights
        else:
            method = "weighted_sum"
        weights = legacy_weights

    _validate_signals(signals)

    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unknown blend method '{method}'. Allowed: {sorted(ALLOWED_METHODS)}")

    logger.debug("Blending method=%s rows=%d cols=%d", method, len(signals), len(signals.columns))

    if method == "weighted_sum":
        valid_weights = _validate_weights(signals.columns, weights)
        composite = _weighted_sum(signals, valid_weights)

    elif method == "voting":
        if ((signals < -1) | (signals > 1)).to_numpy().any():
            logger.warning("voting received values outside [-1, 1]; treating as raw vote strengths")
        composite = signals.sum(axis=1).clip(-1, 1)

    else:  # regime_weighted
        valid_weights = _validate_weights(signals.columns, weights)
        active_regime = _resolve_regime(regime, regime_probs)
        if active_regime is None:
            raise ValueError("regime is required for method='regime_weighted'")

        boosts = regime_boosts
        if boosts is None:
            logger.warning(
                "regime_boosts not provided; using DEFAULT_REGIME_BOOSTS fallback (deprecated behavior)"
            )
            boosts = DEFAULT_REGIME_BOOSTS

        if not isinstance(boosts, dict):
            raise ValueError("regime_boosts must be a dict mapping regime to indicator boosts")

        boosted_weights = dict(valid_weights)
        regime_map = boosts.get(active_regime, {})
        for indicator, boost in regime_map.items():
            if indicator in boosted_weights:
                boosted_weights[indicator] *= float(boost)

        total = sum(boosted_weights.values())
        if np.isclose(total, 0.0):
            raise ValueError("boosted weights sum cannot be zero")

        boosted_weights = {k: v / total for k, v in boosted_weights.items()}
        logger.debug("Applied regime boosts regime=%s boosted_weights=%s", active_regime, boosted_weights)
        composite = _weighted_sum(signals, boosted_weights)

    return pd.Series(composite, index=signals.index, name="composite_signal")
