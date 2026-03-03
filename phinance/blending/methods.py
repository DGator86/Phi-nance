"""
phinance.blending.methods
==========================

Pure blend-method implementations.

Each function accepts a signal DataFrame plus weights and returns a
composite pd.Series in roughly [-1, 1].

Available methods
-----------------
  weighted_sum(signals, weights)              — Linear weighted average
  voting(signals, weights, threshold=0.1)     — Majority-vote across signals
  regime_weighted(signals, weights,
                  regime_probs, boost_map)    — Regime-boosted weights
  phiai_chooses(signals, weights)             — Placeholder (→ weighted_sum)

Constants
---------
  BLEND_METHODS — list of valid method name strings
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from phinance.blending.weights import (
    normalise_weights,
    regime_adjusted_weights,
)

BLEND_METHODS = [
    "weighted_sum",
    "regime_weighted",
    "voting",
    "phiai_chooses",
]


# ── Weighted Sum ──────────────────────────────────────────────────────────────


def weighted_sum(
    signals: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """Linear weighted average of signal columns.

    Parameters
    ----------
    signals : DataFrame — columns = indicator names, rows = bars
    weights : dict — ``{indicator_name: weight}``

    Returns
    -------
    pd.Series — composite signal
    """
    cols = list(signals.columns)
    w = normalise_weights(weights, cols)
    out = pd.Series(0.0, index=signals.index)
    for c in cols:
        out = out + signals[c].fillna(0) * w.get(c, 0)
    return out


# ── Voting ────────────────────────────────────────────────────────────────────


def voting(
    signals: pd.DataFrame,
    weights: Dict[str, float],
    threshold: float = 0.1,
) -> pd.Series:
    """Majority vote: each indicator votes -1, 0, or +1.

    Votes are optionally weighted; the result is clipped to [-1, 1].

    Parameters
    ----------
    signals   : DataFrame
    weights   : dict — weight each vote (uniform if not supplied)
    threshold : float — signal magnitude to count as a vote (default 0.1)

    Returns
    -------
    pd.Series — clipped majority signal
    """
    cols = list(signals.columns)
    w = normalise_weights(weights, cols)
    weighted_vote = pd.Series(0.0, index=signals.index)
    for c in cols:
        s = signals[c].fillna(0)
        vote = pd.Series(
            np.where(s > threshold, 1, np.where(s < -threshold, -1, 0)),
            index=signals.index,
        )
        weighted_vote = weighted_vote + vote * w.get(c, 1.0 / len(cols))
    return weighted_vote.clip(-1, 1)


# ── Regime Weighted ───────────────────────────────────────────────────────────

# Default regime→indicator boost table.
# Boost factors (>1.0 = amplify, <1.0 = dampen) per indicator-regime pairing.
# Sources: academic indicator-regime affinity research + original phi blender.
_DEFAULT_BOOST_MAP: Dict[str, Dict[str, float]] = {

    # ── Mean-reversion oscillators (thrive in range-bound, low-vol markets) ──
    "RSI": {
        "RANGE": 1.4, "LOWVOL": 1.2,
        "TREND_UP": 1.0, "TREND_DN": 1.0,
        "BREAKOUT_UP": 0.9, "BREAKOUT_DN": 0.9, "HIGHVOL": 0.8,
    },
    "Bollinger": {
        "RANGE": 1.5, "LOWVOL": 1.3,
        "TREND_UP": 0.9, "TREND_DN": 0.9,
        "BREAKOUT_UP": 0.7, "BREAKOUT_DN": 0.7, "HIGHVOL": 0.8,
    },
    "Mean Reversion": {
        "RANGE": 1.6, "LOWVOL": 1.3,
        "TREND_UP": 0.5, "TREND_DN": 0.5,
        "BREAKOUT_UP": 0.4, "BREAKOUT_DN": 0.4, "HIGHVOL": 0.7,
    },
    "Stochastic": {
        "RANGE": 1.4, "LOWVOL": 1.2,
        "TREND_UP": 1.0, "TREND_DN": 1.0,
        "BREAKOUT_UP": 0.8, "BREAKOUT_DN": 0.8, "HIGHVOL": 0.8,
    },
    "Williams %R": {
        "RANGE": 1.3, "LOWVOL": 1.2,
        "TREND_UP": 1.0, "TREND_DN": 1.0,
        "BREAKOUT_UP": 0.8, "BREAKOUT_DN": 0.8,
    },
    "CCI": {
        "RANGE": 1.3, "LOWVOL": 1.1,
        "TREND_UP": 1.0, "TREND_DN": 1.0,
        "HIGHVOL": 0.9, "BREAKOUT_UP": 0.9, "BREAKOUT_DN": 0.9,
    },
    "VWAP": {
        "RANGE": 1.5, "LOWVOL": 1.3,
        "TREND_UP": 0.7, "TREND_DN": 0.7,
        "BREAKOUT_UP": 0.5, "BREAKOUT_DN": 0.5,
    },

    # ── Trend-following / momentum (thrive in trending, high-vol markets) ────
    "MACD": {
        "TREND_UP": 1.5, "TREND_DN": 1.5,
        "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "HIGHVOL": 1.2, "RANGE": 0.5, "LOWVOL": 0.6,
    },
    "Dual SMA": {
        "TREND_UP": 1.5, "TREND_DN": 1.5,
        "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "HIGHVOL": 1.1, "RANGE": 0.4, "LOWVOL": 0.5,
    },
    "EMA Cross": {
        "TREND_UP": 1.4, "TREND_DN": 1.4,
        "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "HIGHVOL": 1.1, "RANGE": 0.5, "LOWVOL": 0.6,
    },
    "Breakout": {
        "BREAKOUT_UP": 1.6, "BREAKOUT_DN": 1.6,
        "TREND_UP": 1.3, "TREND_DN": 1.3,
        "HIGHVOL": 1.2, "RANGE": 0.5, "LOWVOL": 0.6,
    },
    "PSAR": {
        "TREND_UP": 1.5, "TREND_DN": 1.5,
        "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "HIGHVOL": 1.2, "RANGE": 0.4, "LOWVOL": 0.5,
    },
    "OBV": {
        "TREND_UP": 1.4, "TREND_DN": 1.4,
        "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "RANGE": 0.7, "LOWVOL": 0.8,
    },

    # ── Volatility-regime indicator (neutral signal, regime context) ─────────
    "ATR": {
        "HIGHVOL": 1.4, "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3,
        "TREND_UP": 1.1, "TREND_DN": 1.1,
        "RANGE": 0.7, "LOWVOL": 0.6,
    },

    # ── Baseline (always-on, no regime bias) ─────────────────────────────────
    "Buy & Hold": {},
}

REGIME_INDICATOR_BOOST = _DEFAULT_BOOST_MAP  # Public alias


def regime_weighted(
    signals: pd.DataFrame,
    weights: Dict[str, float],
    regime_probs: pd.DataFrame,
    boost_map: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    """Regime-boosted weighted sum.

    Indicator weights are amplified for regimes where they excel and
    dampened for regimes where they historically underperform.

    Parameters
    ----------
    signals      : DataFrame
    weights      : dict — base weights
    regime_probs : DataFrame — columns = regime names, index = bar datetime
    boost_map    : dict, optional — ``{indicator: {regime: factor}}``.
                   Defaults to ``REGIME_INDICATOR_BOOST``.

    Returns
    -------
    pd.Series — composite signal
    """
    _map = boost_map if boost_map is not None else _DEFAULT_BOOST_MAP
    cols = list(signals.columns)
    base_w = normalise_weights(weights, cols)
    adj = regime_adjusted_weights(base_w, regime_probs, _map, signals.index)
    total_w = adj.sum(axis=1).replace(0, 1.0)
    filled = signals[cols].fillna(0.0)
    return (filled * adj).sum(axis=1) / total_w


# ── PhiAI chooses ─────────────────────────────────────────────────────────────


def phiai_chooses(
    signals: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """Placeholder: PhiAI-selected method currently falls back to weighted_sum."""
    return weighted_sum(signals, weights)
