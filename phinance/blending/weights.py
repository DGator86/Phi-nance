"""
phinance.blending.weights
=========================

Weight calculation and normalisation helpers for signal blending.

Functions
---------
  normalise_weights(weights, names)    — Ensure weights sum to 1.0
  equal_weights(names)                 — Return 1/N weights for all names
  boost_weights(weights, boosts)       — Apply multiplicative boosts
  regime_adjusted_weights(w, regime_probs, boost_map, index) — Vectorised
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def normalise_weights(
    weights: Dict[str, float],
    names: List[str],
) -> Dict[str, float]:
    """Normalise a weight dict so the subset matching *names* sums to 1.

    Parameters
    ----------
    weights : dict — raw weight values (may not sum to 1)
    names   : list — indicator names to include

    Returns
    -------
    dict — normalised weights for *names* only
    """
    w = {n: float(weights.get(n, 1.0 / max(len(names), 1))) for n in names}
    total = sum(w.values())
    if total <= 0:
        total = 1.0
    return {n: v / total for n, v in w.items()}


def equal_weights(names: List[str]) -> Dict[str, float]:
    """Return equal (1/N) weights for all indicator *names*.

    Parameters
    ----------
    names : list — indicator names

    Returns
    -------
    dict — ``{name: 1/N}``
    """
    n = max(len(names), 1)
    return {name: 1.0 / n for name in names}


def boost_weights(
    weights: Dict[str, float],
    boosts: Dict[str, float],
) -> Dict[str, float]:
    """Apply multiplicative boosts then re-normalise.

    Parameters
    ----------
    weights : dict — base weights
    boosts  : dict — ``{name: multiplier}``

    Returns
    -------
    dict — boosted and re-normalised weights
    """
    boosted = {n: w * float(boosts.get(n, 1.0)) for n, w in weights.items()}
    total = sum(boosted.values()) or 1.0
    return {n: v / total for n, v in boosted.items()}


def regime_adjusted_weights(
    base_weights: Dict[str, float],
    regime_probs: pd.DataFrame,
    boost_map: Dict[str, Dict[str, float]],
    index: pd.Index,
) -> pd.DataFrame:
    """Compute per-bar regime-adjusted weights.

    Parameters
    ----------
    base_weights : dict — static base weights per indicator
    regime_probs : DataFrame — regime probability columns aligned to bar index
    boost_map    : dict — ``{indicator: {regime: boost_factor}}``
    index        : pd.Index — target bar index

    Returns
    -------
    pd.DataFrame — shape (n_bars, n_indicators), weights per bar

    Notes
    -----
    For each bar b and indicator i:
      adj_weight[b,i] = base_weight[i] * (1 + sum_r(boost_map[i][r] * prob_r[b]))
    Weights are NOT re-normalised here — that is left to the blend methods.
    """
    names = list(base_weights.keys())
    rp = regime_probs.reindex(index).ffill().bfill()
    uniform = 1.0 / max(len(rp.columns), 1)
    rp = rp.fillna(uniform)

    adj = pd.DataFrame(index=index, columns=names, dtype=float)
    for name in names:
        base_w = base_weights[name]
        ind_boosts = boost_map.get(name, {})
        boost_series = pd.Series(1.0, index=index)
        for regime, factor in ind_boosts.items():
            if regime in rp.columns:
                boost_series = boost_series + rp[regime] * factor
        adj[name] = base_w * boost_series.clip(lower=0.3)

    return adj
