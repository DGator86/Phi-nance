"""
Indicator Blending Engine
=========================

Provides :func:`blend_signals`, which combines multiple indicator signal columns
into a single composite signal using one of four methods:

- ``weighted_sum`` — linear weighted average of all indicator signals.
- ``voting`` — majority-vote across indicators (each votes -1, 0, or +1).
- ``regime_weighted`` — weights boosted/penalised per market regime using
  ``REGIME_INDICATOR_BOOST`` (vectorized implementation).
- ``phiai_chooses`` — placeholder; falls back to ``weighted_sum``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


BLEND_METHODS = ["weighted_sum", "regime_weighted", "voting", "phiai_chooses"]

# Regime bins from regime_engine
REGIME_BINS = ["TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP", "BREAKOUT_DN", "EXHAUST_REV", "LOWVOL", "HIGHVOL"]

# Indicator -> regime boost: which regimes favor this indicator (boost factor)
# Momentum: TREND_UP/DN, BREAKOUT_UP/DN. Mean reversion: RANGE. RSI/Bollinger: RANGE + TREND
REGIME_INDICATOR_BOOST = {
    "RSI": {"TREND_UP": 1.2, "TREND_DN": 1.2, "RANGE": 1.3, "BREAKOUT_UP": 1.1, "BREAKOUT_DN": 1.1},
    "MACD": {"TREND_UP": 1.5, "TREND_DN": 1.5, "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3, "RANGE": 0.6},
    "Bollinger": {"RANGE": 1.4, "TREND_UP": 1.0, "TREND_DN": 1.0, "LOWVOL": 1.2},
    "Dual SMA": {"TREND_UP": 1.5, "TREND_DN": 1.5, "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3, "RANGE": 0.5},
    "Mean Reversion": {"RANGE": 1.6, "LOWVOL": 1.2, "TREND_UP": 0.5, "TREND_DN": 0.5},
    "Breakout": {"BREAKOUT_UP": 1.5, "BREAKOUT_DN": 1.5, "TREND_UP": 1.2, "TREND_DN": 1.2, "RANGE": 0.6},
    "Buy & Hold": {},  # No regime preference
    # VWAP is a mean-reversion signal — strongest in ranging/low-vol regimes,
    # weaker in trends where price can sustain deviation for extended periods.
    "VWAP": {"RANGE": 1.5, "LOWVOL": 1.3, "EXHAUST_REV": 1.2, "TREND_UP": 0.7, "TREND_DN": 0.7, "BREAKOUT_UP": 0.5, "BREAKOUT_DN": 0.5},
}


def blend_signals(
    signals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_sum",
    regime_probs: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Blend multiple indicator signal columns into a single composite.

    Parameters
    ----------
    signals : DataFrame with columns = indicator names
    weights : optional dict of {indicator_name: weight}
    method : weighted_sum | regime_weighted | voting | phiai_chooses
    regime_probs : optional regime probabilities (for regime_weighted)

    Returns
    -------
    pd.Series — composite signal, same index as signals
    """
    if signals.empty or signals.columns.empty:
        return pd.Series(dtype=float)

    cols = list(signals.columns)
    w = weights or {c: 1.0 / len(cols) for c in cols}
    wsum = sum(w.get(c, 0) for c in cols)
    if wsum <= 0:
        wsum = 1.0
    for c in cols:
        w.setdefault(c, 1.0 / len(cols))

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

    if method == "regime_weighted" and regime_probs is not None:
        # Vectorized: align regime_probs to signals index, compute per-indicator
        # adjusted weights using numpy, then compute weighted sum in one pass.
        rp_aligned = regime_probs.reindex(signals.index).ffill().bfill()
        # Fill any remaining NaNs with uniform distribution
        rp_aligned = rp_aligned.fillna(1.0 / max(len(rp_aligned.columns), 1))

        adj_weights = pd.DataFrame(index=signals.index, columns=cols, dtype=float)
        for c in cols:
            base_w = w.get(c, 1.0 / len(cols))
            boost_map = REGIME_INDICATOR_BOOST.get(c, {})
            # boost = 1 + sum(boost_factor * regime_prob) for each regime
            boost = pd.Series(1.0, index=signals.index)
            for regime, factor in boost_map.items():
                if regime in rp_aligned.columns:
                    boost = boost + rp_aligned[regime] * factor
            adj_weights[c] = base_w * boost.clip(lower=0.3)

        total_w = adj_weights.sum(axis=1).replace(0, 1.0)
        filled = signals[cols].fillna(0.0)
        out = (filled * adj_weights).sum(axis=1) / total_w
        return out.fillna(0)

    if method == "phiai_chooses":
        # Placeholder: same as weighted sum; PhiAI would optimize weights
        return blend_signals(signals, weights, "weighted_sum", None)

    # Unknown method — fall back to weighted_sum
    logger.warning("blend_signals: unknown method %r; falling back to weighted_sum", method)
    return blend_signals(signals, weights, "weighted_sum", None)
