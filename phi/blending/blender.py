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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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

    result = sum(signals[c] * w[c] for c in cols)
    return np.tanh(result).rename("blend")


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

    total_w = sum(w.values()) + 1e-10
    votes   = sum(
        np.sign(signals[c].where(signals[c].abs() > 0.05, 0.0)) * w[c] / total_w
        for c in cols
    )
    return np.tanh(votes * 2.0).rename("blend")


def blend_regime_weighted(
    signals: pd.DataFrame,
    regime_weights: Dict[str, Dict[str, float]],
    regime_probs: Optional[pd.DataFrame] = None,
    default_weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Regime-weighted blend.

    Parameters
    ----------
    signals        : DataFrame of indicator signals
    regime_weights : {regime_name: {indicator_name: weight}}
    regime_probs   : DataFrame of regime probabilities (same index as signals)
    default_weights: fallback weights if no regime info available

    Returns
    -------
    pd.Series in [-1, +1]
    """
    if signals.empty:
        return pd.Series(dtype=float)

    if regime_probs is None or regime_probs.empty:
        return blend_weighted_sum(signals, default_weights)

    # Align index
    aligned_probs = regime_probs.reindex(signals.index).ffill().fillna(0.0)

    result = pd.Series(0.0, index=signals.index)
    cols   = list(signals.columns)

    for regime, rw in regime_weights.items():
        if regime not in aligned_probs.columns:
            continue
        prob_series = aligned_probs[regime]
        total_rw = sum(rw.get(c, 1.0) for c in cols) + 1e-10
        regime_signal = sum(
            signals[c] * (rw.get(c, 1.0) / total_rw)
            for c in cols
        )
        result = result + prob_series * regime_signal

    return np.tanh(result).rename("blend")


def blend_phiai(
    signals: pd.DataFrame,
    metric_scores: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    PhiAI blend: weight each indicator by its historical metric score.

    Parameters
    ----------
    signals       : DataFrame of indicator signals
    metric_scores : {indicator_name: score} — higher = better

    Returns
    -------
    pd.Series in [-1, +1]
    """
    if signals.empty:
        return pd.Series(dtype=float)

    if not metric_scores:
        return blend_weighted_sum(signals)

    cols = list(signals.columns)
    scores = {c: max(metric_scores.get(c, 0.0), 0.0) for c in cols}
    total = sum(scores.values()) + 1e-10

    if total < 1e-8:
        return blend_weighted_sum(signals)

    weights = {c: scores[c] / total for c in cols}
    return blend_weighted_sum(signals, weights)


# ─────────────────────────────────────────────────────────────────────────────
# Main blender interface
# ─────────────────────────────────────────────────────────────────────────────

class Blender:
    """
    Stateful blender that holds the chosen mode and parameters.

    Usage
    -----
    blender = Blender(mode='weighted_sum', weights={'rsi': 0.4, 'macd': 0.6})
    blended = blender.blend(signals_df)
    """

    def __init__(
        self,
        mode: str = "weighted_sum",
        weights: Optional[Dict[str, float]] = None,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        metric_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        self.mode           = mode
        self.weights        = weights or {}
        self.regime_weights = regime_weights or {}
        self.metric_scores  = metric_scores or {}

    def blend(
        self,
        signals: pd.DataFrame,
        regime_probs: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        if self.mode == "weighted_sum":
            return blend_weighted_sum(signals, self.weights or None)

        elif self.mode == "voting":
            return blend_voting(signals, self.weights or None)

        elif self.mode == "regime_weighted":
            return blend_regime_weighted(
                signals,
                self.regime_weights,
                regime_probs,
                self.weights or None,
            )

        elif self.mode == "phiai":
            return blend_phiai(signals, self.metric_scores or None)

        else:
            return blend_weighted_sum(signals, self.weights or None)

    def blend_preview(self, signals: pd.DataFrame, n_bars: int = 100) -> pd.Series:
        """Return last n_bars of blended signal for quick preview."""
        return self.blend(signals.tail(n_bars))


def default_regime_weights(indicator_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Build sensible default regime-weight matrix for a given list of indicators.
    Trend indicators get higher weight in TREND regimes, oscillators in RANGE.
    """
    trend_indicators   = {"macd", "dual_sma", "ema_crossover", "momentum", "roc", "adx"}
    range_indicators   = {"rsi", "stochastic", "bollinger", "range_pos", "cmf", "vwap_dev"}
    vol_indicators     = {"atr_ratio"}

    regimes = [
        "TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP", "BREAKOUT_DN",
        "EXHAUST_REV", "LOWVOL", "HIGHVOL",
    ]

    weights: Dict[str, Dict[str, float]] = {}
    for regime in regimes:
        rw: Dict[str, float] = {}
        for ind in indicator_names:
            if ind in trend_indicators:
                rw[ind] = 1.5 if "TREND" in regime or "BREAKOUT" in regime else 0.5
            elif ind in range_indicators:
                rw[ind] = 1.5 if regime in ("RANGE", "EXHAUST_REV") else 0.7
            elif ind in vol_indicators:
                rw[ind] = 1.3 if regime in ("HIGHVOL", "BREAKOUT_UP", "BREAKOUT_DN") else 0.8
            else:
                rw[ind] = 1.0
        weights[regime] = rw
    return weights
