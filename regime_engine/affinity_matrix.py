"""
Affinity Matrix — Interface 4: Regime Nodes → Indicator Weights.

Implements the entropy-weighted affinity blending function:

  w_affinity(j) = Σ_L  certainty(L) · [Σ_{n∈L} P_t(n) · C(n,j)]

where:
  certainty(L) = 1 - H(P_L) / log(N_L)     (level-specific certainty)
  C(n, j)      = affinity of regime node n for indicator j
  P_t(n)       = linear probability of node n at bar t

Design notes
------------
- The INDICATOR_AFFINITY table has rows for all KPCOFGS nodes that have
  meaningful differentiation. Missing nodes default to 0 (no vote).
- Family ALN has values > 1.0 because it is a conviction *amplifier*, not
  just another regime vote. It multiplicatively boosts when all TFs agree.
- When Kingdom entropy is low (one state dominates), Kingdom-level affinities
  drive weights. When Kingdom is uncertain, lower levels contribute more.
- The final weight is blended with the existing species-pattern validity
  weight in expert_registry.py: w_final = (1-β)·w_validity + β·w_affinity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────────────────
# Indicator column ordering (used to index the matrix)
# ──────────────────────────────────────────────────────────────────────────────
# The matrix is indexed by indicator *name* as it appears in config.yaml.
# Unknown indicators get weight 0 from the affinity table.

AFFINITY_INDICATORS = [
    "ema", "macd", "supertrend",   # trend-following
    "rsi", "bb_mr", "vwap_dev",    # mean-reversion
    "donchian",                     # breakout
    "stochastic", "cmf",            # oscillators / volume
    "roc", "atr_ratio",             # momentum / vol-ratio
    "range_pos", "momentum",        # price structure
    "vol_profile_dev",              # microstructure
]

# ──────────────────────────────────────────────────────────────────────────────
# Affinity table: node → {indicator_name: affinity_weight}
# ──────────────────────────────────────────────────────────────────────────────
# Source: Framework I/O Map, Interface 4 condition matrix.
# Nodes not listed contribute 0.0 affinity for all indicators.

_AFFINITY_TABLE: Dict[str, Dict[str, float]] = {
    # ── KINGDOM ──────────────────────────────────────────────────────
    "DIR": {
        "ema": 0.9, "macd": 0.8, "supertrend": 0.9,
        "rsi": 0.2, "bb_mr": 0.2, "vwap_dev": 0.3,
        "donchian": 0.6, "stochastic": 0.3, "cmf": 0.5,
        "roc": 0.7, "atr_ratio": 0.5, "range_pos": 0.4,
        "momentum": 0.8, "vol_profile_dev": 0.4,
    },
    "NDR": {
        "ema": 0.2, "macd": 0.3, "supertrend": 0.2,
        "rsi": 0.9, "bb_mr": 0.9, "vwap_dev": 0.8,
        "donchian": 0.2, "stochastic": 0.8, "cmf": 0.5,
        "roc": 0.2, "atr_ratio": 0.4, "range_pos": 0.7,
        "momentum": 0.2, "vol_profile_dev": 0.6,
    },
    "TRN": {
        "ema": 0.6, "macd": 0.6, "supertrend": 0.7,
        "rsi": 0.1, "bb_mr": 0.1, "vwap_dev": 0.2,
        "donchian": 0.9, "stochastic": 0.3, "cmf": 0.6,
        "roc": 0.5, "atr_ratio": 0.7, "range_pos": 0.4,
        "momentum": 0.5, "vol_profile_dev": 0.5,
    },
    # ── PHYLUM ───────────────────────────────────────────────────────
    "LV": {
        "ema": 0.5, "macd": 0.5, "supertrend": 0.5,
        "rsi": 0.8, "bb_mr": 0.9, "vwap_dev": 0.8,
        "donchian": 0.6, "stochastic": 0.7, "cmf": 0.5,
        "roc": 0.4, "atr_ratio": 0.2, "range_pos": 0.7,
        "momentum": 0.3, "vol_profile_dev": 0.6,
    },
    "HV": {
        "ema": 0.7, "macd": 0.7, "supertrend": 0.8,
        "rsi": 0.2, "bb_mr": 0.2, "vwap_dev": 0.3,
        "donchian": 0.7, "stochastic": 0.3, "cmf": 0.4,
        "roc": 0.6, "atr_ratio": 0.9, "range_pos": 0.4,
        "momentum": 0.6, "vol_profile_dev": 0.5,
    },
    # ── CLASS (selected) ─────────────────────────────────────────────
    "PT": {
        "ema": 1.0, "macd": 0.9, "supertrend": 0.9,
        "rsi": 0.2, "bb_mr": 0.2, "vwap_dev": 0.3,
        "donchian": 0.5, "roc": 0.8, "momentum": 0.9,
        "atr_ratio": 0.4, "range_pos": 0.5,
    },
    "PX": {
        "ema": 0.8, "macd": 0.8, "supertrend": 0.9,
        "rsi": 0.1, "bb_mr": 0.1, "vwap_dev": 0.2,
        "donchian": 0.8, "roc": 0.7, "momentum": 0.8,
        "atr_ratio": 0.8, "range_pos": 0.4,
    },
    "TE": {
        "ema": 0.3, "macd": 0.3, "supertrend": 0.4,
        "rsi": 0.5, "bb_mr": 0.5, "vwap_dev": 0.5,
        "donchian": 0.3, "atr_ratio": 0.7, "range_pos": 0.5,
    },
    "BR": {
        "ema": 0.2, "macd": 0.2, "supertrend": 0.2,
        "rsi": 1.0, "bb_mr": 1.0, "vwap_dev": 0.9,
        "donchian": 0.1, "stochastic": 0.9, "range_pos": 0.8,
    },
    "RR": {
        "ema": 0.2, "macd": 0.2, "supertrend": 0.2,
        "rsi": 0.9, "bb_mr": 0.9, "vwap_dev": 0.8,
        "donchian": 0.1, "stochastic": 0.8, "range_pos": 0.7,
    },
    "SR": {
        "ema": 0.5, "macd": 0.5, "supertrend": 0.6,
        "rsi": 0.2, "bb_mr": 0.2, "vwap_dev": 0.2,
        "donchian": 1.0, "atr_ratio": 0.7, "vol_profile_dev": 0.7,
    },
    "RB": {
        "ema": 0.7, "macd": 0.7, "supertrend": 0.8,
        "rsi": 0.1, "bb_mr": 0.1, "vwap_dev": 0.2,
        "donchian": 1.0, "atr_ratio": 0.6, "momentum": 0.7,
    },
    "FB": {
        "ema": 0.2, "macd": 0.2, "supertrend": 0.2,
        "rsi": 0.6, "bb_mr": 0.6, "vwap_dev": 0.6,
        "donchian": 0.3, "atr_ratio": 0.5,
    },
    # ── ORDER ────────────────────────────────────────────────────────
    "AGC": {
        "ema": 0.9, "macd": 0.9, "supertrend": 0.9,
        "rsi": 0.1, "bb_mr": 0.1, "vwap_dev": 0.2,
        "donchian": 0.8, "roc": 0.8, "momentum": 0.9,
    },
    "RVP": {
        "ema": 0.2, "macd": 0.2, "supertrend": 0.2,
        "rsi": 0.9, "bb_mr": 0.9, "vwap_dev": 0.9,
        "donchian": 0.1, "stochastic": 0.8,
    },
    "ABS": {
        "ema": 0.3, "macd": 0.3, "supertrend": 0.3,
        "rsi": 0.5, "bb_mr": 0.5, "vwap_dev": 0.6,
        "donchian": 0.2, "cmf": 0.7, "vol_profile_dev": 0.7,
    },
    "EXH": {
        "ema": 0.2, "macd": 0.2, "supertrend": 0.2,
        "rsi": 0.4, "bb_mr": 0.4, "vwap_dev": 0.4,
        "donchian": 0.2, "atr_ratio": 0.8, "range_pos": 0.5,
    },
    # ── FAMILY ───────────────────────────────────────────────────────
    # ALN > 1.0 intentional — cross-TF agreement is a conviction amplifier
    "ALN": {
        "ema": 1.1, "macd": 1.0, "supertrend": 1.1,
        "rsi": 1.1, "bb_mr": 1.0, "vwap_dev": 1.0,
        "donchian": 1.1, "stochastic": 1.0, "cmf": 1.0,
        "roc": 1.0, "atr_ratio": 1.0, "range_pos": 1.0,
        "momentum": 1.0, "vol_profile_dev": 1.0,
    },
    "CT": {
        "ema": 0.6, "macd": 0.6, "supertrend": 0.6,
        "rsi": 0.6, "bb_mr": 0.6, "vwap_dev": 0.6,
        "donchian": 0.6, "momentum": 0.6,
    },
    "CST": {
        "ema": 0.4, "macd": 0.4, "supertrend": 0.4,
        "rsi": 0.7, "bb_mr": 0.7, "vwap_dev": 0.7,
        "donchian": 0.8, "atr_ratio": 0.3, "vol_profile_dev": 0.7,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Level groupings for entropy-weighted blending
# ──────────────────────────────────────────────────────────────────────────────

_LEVEL_GROUPS: Dict[str, List[str]] = {
    "kingdom": ["DIR", "NDR", "TRN"],
    "phylum":  ["LV", "NV", "HV"],
    "class_":  ["PT", "PX", "TE", "BR", "RR", "AR", "SR", "RB", "FB"],
    "order":   ["AGC", "RVP", "ABS", "EXH"],
    "family":  ["ALN", "CT", "CST"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────

def entropy_certainty(
    node_log_probs: pd.DataFrame,
    nodes: List[str],
) -> pd.Series:
    """
    Compute per-bar level certainty: 1 - H(P_level) / log(N_level).

    Returns 0 when uniform (maximum uncertainty), 1 when perfectly concentrated.

    Parameters
    ----------
    node_log_probs : DataFrame of log-probabilities (any subset of nodes)
    nodes          : list of node names forming this sibling group

    Returns
    -------
    pd.Series of shape (T,), values in [0, 1]
    """
    present = [n for n in nodes if n in node_log_probs.columns]
    if not present:
        return pd.Series(0.5, index=node_log_probs.index)

    lp = node_log_probs[present].values.astype(np.float64)
    p = np.exp(np.clip(lp, -500, 0))
    row_sums = p.sum(axis=1, keepdims=True).clip(min=1e-15)
    p = p / row_sums

    h = -(p * np.log(p + 1e-15)).sum(axis=1)
    h_max = np.log(max(len(present), 2))
    certainty = 1.0 - np.clip(h / h_max, 0.0, 1.0)
    return pd.Series(certainty, index=node_log_probs.index)


def compute_entropy_weighted_affinity(
    node_log_probs: pd.DataFrame,
    indicator_names: List[str],
) -> pd.DataFrame:
    """
    Compute per-bar affinity-based indicator weights using entropy weighting.

    w_affinity(j) = Σ_L  certainty(L) · [Σ_{n∈L} P_t(n) · C(n,j)]

    Parameters
    ----------
    node_log_probs  : (T, num_nodes) log-prob DataFrame from ProbabilityField
    indicator_names : list of indicator names to compute weights for

    Returns
    -------
    pd.DataFrame of shape (T, len(indicator_names)) — raw affinity weights
    Values are NOT clipped to [0,1] — ALN can produce values > 1.
    """
    T = len(node_log_probs)
    # Accumulator: (T, num_indicators)
    w_accum = np.zeros((T, len(indicator_names)), dtype=np.float64)
    ind_idx = {name: i for i, name in enumerate(indicator_names)}

    for level, nodes in _LEVEL_GROUPS.items():
        cert = entropy_certainty(node_log_probs, nodes).values  # (T,)

        # Linear probabilities for nodes in this level
        present = [n for n in nodes if n in node_log_probs.columns]
        if not present:
            continue
        lp = node_log_probs[present].values.astype(np.float64)
        p = np.exp(np.clip(lp, -500, 0))  # (T, |present|)

        # Weighted sum: Σ_{n∈L} P_t(n) · C(n,j)
        level_vote = np.zeros((T, len(indicator_names)), dtype=np.float64)
        for k, node in enumerate(present):
            node_affinities = _AFFINITY_TABLE.get(node, {})
            for ind_name, aff_weight in node_affinities.items():
                ji = ind_idx.get(ind_name)
                if ji is not None:
                    level_vote[:, ji] += p[:, k] * aff_weight

        # Scale by level certainty and accumulate
        w_accum += cert[:, None] * level_vote

    return pd.DataFrame(
        w_accum,
        index=node_log_probs.index,
        columns=indicator_names,
    )
