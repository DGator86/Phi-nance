"""
phi.blending.blender — Signal Blending Engine
===============================================
Blend modes:
  weighted_sum     — sum(w_i * signal_i)
  voting           — majority-vote of sign(signal_i)
  regime_weighted  — weights per MFT regime × probability
  phiai            — PhiAI chooses best indicator per bar
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


BLEND_MODES = ["weighted_sum", "voting", "regime_weighted", "phiai"]
BLEND_LABELS = {
    "weighted_sum":    "Weighted Sum",
    "voting":          "Voting (Majority)",
    "regime_weighted": "Regime-Weighted",
    "phiai":           "PhiAI Chooses",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core blending functions
# ─────────────────────────────────────────────────────────────────────────────

def blend_weighted_sum(
    signals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Weighted sum of signals.

    Parameters
    ----------
    signals : DataFrame, columns = indicator names, values in [-1, +1]
    weights : {indicator_name: weight}; if None, equal weights

    Returns
    -------
    pd.Series in [-1, +1]
    """
    if signals.empty:
        return pd.Series(dtype=float)

    cols = list(signals.columns)
    if weights is None:
        w = {c: 1.0 / len(cols) for c in cols}
    else:
        # Normalize provided weights to sum to 1
        total = sum(weights.get(c, 1.0) for c in cols)
        w = {c: weights.get(c, 1.0) / (total + 1e-10) for c in cols}

    if method == "weighted_sum":
        col_weights = pd.Series({c: w.get(c, 0) / wsum for c in cols})
        return signals[cols].fillna(0).mul(col_weights).sum(axis=1)

    if method == "voting":
        # Each indicator votes -1, 0, or 1; majority wins — fully vectorized.
        filled = signals[cols].fillna(0)
        votes = (filled > 0.1).astype(int) - (filled < -0.1).astype(int)
        return (votes.sum(axis=1) / len(cols)).clip(-1, 1)

    if method == "regime_weighted" and regime_probs is None:
        logger.warning(
            "blend_signals: method='regime_weighted' requested but regime_probs=None; "
            "falling back to weighted_sum. Pass regime probabilities from RegimeEngine "
            "to enable regime-aware blending."
        )
        return blend_signals(signals, weights, "weighted_sum", None)

def blend_voting(
    signals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Weighted majority vote: sign(sum of weighted votes).

    Each indicator votes +1 (if signal > 0.1) or -1 (if < -0.1) or 0.
    Final signal is strength of majority.
    """
    if signals.empty:
        return pd.Series(dtype=float)

    cols   = list(signals.columns)
    n      = len(cols)
    total_w = 0.0

    if weights is None:
        w = {c: 1.0 for c in cols}
    else:
        w = {c: weights.get(c, 1.0) for c in cols}

    # Unknown method — fall back to weighted_sum
    logger.warning("blend_signals: unknown method %r; falling back to weighted_sum", method)
    return blend_signals(signals, weights, "weighted_sum", None)
