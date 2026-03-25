"""Regime-quality metrics beyond raw classification accuracy (trading-oriented)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regime_entropy_normalized(labels: pd.Series) -> float:
    """Normalized Shannon entropy of the regime label distribution in ``[0, 1]``.

    0 = single regime only; 1 = maximum mixing given the number of distinct labels observed.
    """
    s = labels.dropna()
    if s.empty:
        return 0.0
    vc = s.astype(str).value_counts()
    p = (vc / vc.sum()).to_numpy(dtype=float)
    h = float(-np.sum(p * np.log(p + 1e-12)))
    k = len(p)
    if k <= 1:
        return 0.0
    return float(h / np.log(k))


def transition_rate(labels: pd.Series) -> float:
    """Fraction of adjacent bars where the regime label changes."""
    s = labels.dropna().astype(str)
    if len(s) < 2:
        return 0.0
    changes = (s != s.shift(1)).sum() - 1
    return float(max(changes, 0) / (len(s) - 1))


def mean_run_length_bars(labels: pd.Series) -> float:
    """Average length of constant-regime runs (persistence)."""
    s = labels.dropna().astype(str)
    if s.empty:
        return 0.0
    changes = s != s.shift(1)
    run_id = changes.cumsum()
    lengths = s.groupby(run_id).size().to_numpy(dtype=float)
    return float(lengths.mean()) if len(lengths) else 0.0


def per_regime_forward_return_summary(
    close: pd.Series,
    labels: pd.Series,
    horizon: int = 1,
) -> dict[str, Any]:
    """Per-regime next-``horizon`` log-return mean/std/count (rough economic plausibility check)."""
    c = close.reindex(labels.index).astype(float).dropna()
    lab = labels.reindex(c.index).dropna()
    common = c.index.intersection(lab.index)
    if len(common) < horizon + 2:
        return {}
    c = c.reindex(common)
    lab = lab.reindex(common)
    fwd = np.log(c.shift(-horizon) / c).replace([np.inf, -np.inf], np.nan)
    out: dict[str, dict[str, float]] = {}
    for reg in lab.dropna().unique():
        mask = lab == reg
        r = fwd.where(mask).dropna()
        if r.empty:
            continue
        key = str(reg)
        out[key] = {
            "mean": float(r.mean()),
            "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            "n": float(len(r)),
        }
    return out


def summarize_regime_metrics(ohlcv: pd.DataFrame, regime_series: pd.Series) -> dict[str, float]:
    """Single flat dict of floats for logging (MLflow, manifests)."""
    ent = regime_entropy_normalized(regime_series)
    tr = transition_rate(regime_series)
    mrl = mean_run_length_bars(regime_series)
    close = _close_series(ohlcv)
    per = per_regime_forward_return_summary(close, regime_series, horizon=1)
    means = [v["mean"] for v in per.values()] if per else []
    spread = float(max(means) - min(means)) if len(means) >= 2 else 0.0
    return {
        "regime_entropy_norm": ent,
        "regime_transition_rate": tr,
        "regime_mean_run_length": mrl,
        "regime_fwd_return_spread_d1": spread,
    }


def _close_series(df: pd.DataFrame) -> pd.Series:
    cols = {str(c).lower(): c for c in df.columns}
    if "close" not in cols:
        raise ValueError("OHLCV must contain a 'close' column")
    return df[cols["close"]].astype(float)
