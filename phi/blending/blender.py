"""
Indicator Blending Engine
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


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
        out = pd.Series(0.0, index=signals.index)
        for c in cols:
            s = signals[c].fillna(0)
            out = out + s * (w.get(c, 0) / wsum)
        return out

    if method == "voting":
        # Each indicator votes -1, 0, or 1; majority wins
        votes = pd.DataFrame(index=signals.index)
        for c in cols:
            s = signals[c].fillna(0)
            votes[c] = np.where(s > 0.1, 1, np.where(s < -0.1, -1, 0))
        majority = votes.sum(axis=1) / len(cols)
        return majority.clip(-1, 1)

    if method == "regime_weighted" and regime_probs is not None:
        # Scale weights by regime appropriateness per bar
        rp_aligned = regime_probs.reindex(signals.index).ffill().bfill()
        out = pd.Series(0.0, index=signals.index)
        for idx in signals.index:
            if idx not in rp_aligned.index or rp_aligned.loc[idx].isna().all():
                rp = pd.Series(1.0 / 8, index=REGIME_BINS)
            else:
                rp = rp_aligned.loc[idx]
            total_w = 0.0
            weighted_sum = 0.0
            for c in cols:
                base_w = w.get(c, 1.0 / len(cols))
                boost = 1.0
                for regime, prob in rp.items():
                    if regime in REGIME_INDICATOR_BOOST.get(c, {}):
                        boost += REGIME_INDICATOR_BOOST[c][regime] * prob
                adj_w = base_w * max(0.3, boost)
                total_w += adj_w
                s_val = signals.loc[idx, c] if pd.notna(signals.loc[idx, c]) else 0.0
                weighted_sum += s_val * adj_w
            if total_w > 0:
                out.loc[idx] = weighted_sum / total_w
        return out.fillna(0)

    if method == "phiai_chooses":
        # Placeholder: same as weighted sum; PhiAI would optimize weights
        return blend_signals(signals, weights, "weighted_sum", None)

    return blend_signals(signals, weights, "weighted_sum", None)
