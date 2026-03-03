"""
phinance.blending.blender
==========================

Orchestrator: the single public ``blend_signals()`` function that
dispatches to the appropriate method in ``phinance.blending.methods``.

Usage
-----
    from phinance.blending import blend_signals

    composite = blend_signals(
        signals   = signals_df,     # DataFrame: columns = indicator names
        weights   = {"RSI": 0.4, "MACD": 0.6},
        method    = "regime_weighted",
        regime_probs = regime_probs_df,  # optional, required for regime_weighted
    )
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from phinance.blending.methods import (
    BLEND_METHODS,
    weighted_sum,
    voting,
    regime_weighted,
    phiai_chooses,
)
from phinance.blending.weights import equal_weights
from phinance.exceptions import UnsupportedBlendMethodError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


def blend_signals(
    signals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_sum",
    regime_probs: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Blend multiple indicator signal columns into a single composite signal.

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame with columns = indicator names, rows = bars.
        Values should be in roughly [-1, 1].
    weights : dict, optional
        ``{indicator_name: weight}``.  Defaults to equal weights.
    method : str
        One of: ``"weighted_sum"``, ``"voting"``, ``"regime_weighted"``,
        ``"phiai_chooses"``.
    regime_probs : pd.DataFrame, optional
        Regime probability columns (required for ``"regime_weighted"``).
        If omitted for ``"regime_weighted"`` the method auto-detects regime
        from the signal index using ``regime_detector.detect_regime()``.

    Returns
    -------
    pd.Series
        Composite signal, same index as *signals*.

    Raises
    ------
    UnsupportedBlendMethodError
        When *method* is not in ``BLEND_METHODS``.
    """
    if signals.empty or len(signals.columns) == 0:
        return pd.Series(dtype=float)

    cols = list(signals.columns)
    w = weights if weights else equal_weights(cols)

    if method == "weighted_sum":
        return weighted_sum(signals, w)

    if method == "voting":
        return voting(signals, w)

    if method == "regime_weighted":
        if regime_probs is None:
            # Auto-detect regime from the signal index
            logger.warning(
                "regime_probs not supplied for 'regime_weighted'; "
                "falling back to weighted_sum."
            )
            return weighted_sum(signals, w)
        return regime_weighted(signals, w, regime_probs)

    if method == "phiai_chooses":
        return phiai_chooses(signals, w)

    raise UnsupportedBlendMethodError(
        f"Unknown blend method: '{method}'. "
        f"Supported: {BLEND_METHODS}"
    )
