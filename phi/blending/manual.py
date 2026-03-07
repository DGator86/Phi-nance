"""Manual (non-AI) blend methods."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BlendMethod


def _normalise_weights(signals: pd.DataFrame, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    cols = list(signals.columns)
    if not cols:
        return {}
    w = dict(weights or {c: 1.0 / len(cols) for c in cols})
    for c in cols:
        w.setdefault(c, 1.0 / len(cols))
    total = sum(max(0.0, float(w.get(c, 0.0))) for c in cols)
    if total <= 0:
        return {c: 1.0 / len(cols) for c in cols}
    return {c: max(0.0, float(w.get(c, 0.0))) / total for c in cols}


class WeightedSum(BlendMethod):
    def blend(self, signals: pd.DataFrame, **kwargs) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)
        weights = _normalise_weights(signals, kwargs.get("weights"))
        out = pd.Series(0.0, index=signals.index)
        for col in signals.columns:
            out = out + signals[col].fillna(0.0) * weights[col]
        return out

    def get_params(self) -> dict:
        return {"weights": "dict[str, float]"}


class Voting(BlendMethod):
    def blend(self, signals: pd.DataFrame, **kwargs) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)
        threshold = float(kwargs.get("vote_threshold", 0.1))
        votes = pd.DataFrame(index=signals.index)
        for col in signals.columns:
            s = signals[col].fillna(0.0)
            votes[col] = np.where(s > threshold, 1, np.where(s < -threshold, -1, 0))
        return (votes.sum(axis=1) / max(len(signals.columns), 1)).clip(-1, 1)

    def get_params(self) -> dict:
        return {"vote_threshold": "float"}


class RegimeWeighted(BlendMethod):
    def blend(self, signals: pd.DataFrame, **kwargs) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        regime_probs: Optional[pd.DataFrame] = kwargs.get("regime_probs")
        if regime_probs is None or regime_probs.empty:
            return WeightedSum().blend(signals, **kwargs)

        cols = list(signals.columns)
        weights = _normalise_weights(signals, kwargs.get("weights"))
        rp_aligned = regime_probs.reindex(signals.index).ffill().bfill()
        rp_aligned = rp_aligned.fillna(1.0 / max(len(rp_aligned.columns), 1))

        regime_boosts = kwargs.get("regime_boosts") or {}

        adjusted = pd.DataFrame(index=signals.index, columns=cols, dtype=float)
        for col in cols:
            boost = pd.Series(1.0, index=signals.index)
            indicator_boosts = regime_boosts.get(col, {}) if isinstance(regime_boosts, dict) else {}
            for regime, factor in indicator_boosts.items():
                if regime in rp_aligned.columns:
                    boost = boost + rp_aligned[regime] * factor
            adjusted[col] = weights[col] * boost.clip(lower=0.3)

        norm = adjusted.sum(axis=1).replace(0.0, 1.0)
        return ((signals[cols].fillna(0.0) * adjusted).sum(axis=1) / norm).fillna(0.0)

    def get_params(self) -> dict:
        return {"weights": "dict[str, float]", "regime_probs": "pd.DataFrame"}
