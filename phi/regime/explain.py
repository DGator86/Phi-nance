"""Lightweight explainability helpers (no hard dependency on SHAP)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phi.regime.utils import extract_features


def feature_regime_correlation_proxy(
    ohlcv: pd.DataFrame,
    regime_series: pd.Series,
    *,
    window: int = 20,
) -> dict[str, float]:
    """Absolute Pearson correlation of each feature with a numeric regime code.

    Uses the same rolling feature matrix as detectors. Useful when SHAP is not installed.
    """
    feats = extract_features(ohlcv, window=window)
    lab = regime_series.reindex(feats.index).dropna()
    common = feats.index.intersection(lab.index)
    if len(common) < 10:
        return {}
    codes = pd.Categorical(lab.reindex(common).astype(str)).codes.astype(float)
    out: dict[str, float] = {}
    for col in feats.columns:
        x = feats[col].reindex(common).astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(codes)
        if mask.sum() < 10:
            continue
        xc = x[mask]
        yc = codes[mask]
        if float(np.std(xc)) < 1e-12 or float(np.std(yc)) < 1e-12:
            continue
        r = float(np.corrcoef(xc, yc)[0, 1])
        out[str(col)] = abs(r)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def try_shap_random_forest_summary(
    ohlcv: pd.DataFrame,
    regime_series: pd.Series,
    *,
    window: int = 20,
    max_samples: int = 2000,
) -> dict[str, Any] | None:
    """Optional global SHAP summary using a small RF surrogate (requires ``shap`` + sklearn)."""
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None

    feats = extract_features(ohlcv, window=window).dropna()
    lab = regime_series.reindex(feats.index).dropna()
    common = feats.index.intersection(lab.index)
    if len(common) < 50:
        return None
    X = feats.reindex(common).astype(float).to_numpy()
    y = pd.Categorical(lab.reindex(common).astype(str)).codes
    if len(common) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(common), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    clf = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42)
    clf.fit(X, y)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X[: min(500, len(X))])
    if isinstance(sv, list):
        shap_arr = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        shap_arr = np.abs(sv).mean(axis=0)
    names = list(feats.columns)
    ranked = dict(sorted(zip(names, shap_arr.tolist()), key=lambda t: -t[1]))
    return {"method": "shap_tree_surrogate_rf", "mean_abs_shap_by_feature": ranked}
