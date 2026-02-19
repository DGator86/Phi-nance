"""
Feature Extractor
-----------------
Unified API to convert a Lumibot bars DataFrame into a single-row
pd.DataFrame of regime features ready for ML classifier `.predict()`.

Usage in a strategy:
    from regime_engine.feature_extractor import get_regime_features

    bars = self.get_bars(symbol, 130, timestep="day")
    df   = bars[asset_key].df
    X    = get_regime_features(df)   # shape: (1, n_features)
    direction = classifier.predict(X)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
import os

from regime_engine.features import FeatureEngine

# ---------------------------------------------------------------------------
# Load the default config (features sub-section)
# ---------------------------------------------------------------------------

def _load_default_config() -> dict:
    """Loads features config from config.yaml if present, else uses defaults."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            full_cfg = yaml.safe_load(fh)
        return full_cfg.get("features", _default_features_config())
    return _default_features_config()


def _default_features_config() -> dict:
    """Minimal config if config.yaml is unavailable."""
    return {
        "drift_windows": [15, 30, 60],
        "vol_windows": [30, 120],
        "er_windows": [30, 60, 120],
        "range_expansion_window": 20,
        "autocorr_window": 30,
        "autocorr_lag": 1,
        "flip_rate_window": 20,
        "median_cross_window": 20,
        "impulse_revert_window": 20,
        "impulse_revert_sigma": 2.0,
        "illiquidity_window": 30,
        "volume_zscore_window": 30,
        "entropy_window": 30,
        "entropy_bins": 10,
        "robust_norm_window": 60,
        "mad_floor": 1e-8,
        "mad_small_multiplier": 10.0,
        "ewma_norm_span": 30,
        "dissipation_window": 30,
    }


_ENGINE: FeatureEngine | None = None


def _get_engine() -> FeatureEngine:
    global _ENGINE
    if _ENGINE is None:
        cfg = _load_default_config()
        _ENGINE = FeatureEngine(cfg)
    return _ENGINE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_regime_features(
    bars_df: pd.DataFrame,
    lookback: int = 120,
) -> pd.DataFrame:
    """
    Extract ML-ready regime features from an OHLCV bars DataFrame.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Lumibot bars DataFrame with columns: open, high, low, close, volume.
        Must have at least `lookback` rows for meaningful features.
    lookback : int
        Number of most-recent rows to use for feature computation.
        The returned DataFrame will always have exactly ONE row
        (the final bar's feature snapshot).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all FeatureEngine.FEATURE_COLS as columns.
        Values are robust z-scores.  NaN columns (warmup) are filled with 0.
    """
    if bars_df is None or len(bars_df) < 5:
        # Return a zero-filled row — classifier should treat this as NEUTRAL
        cols = FeatureEngine.FEATURE_COLS
        return pd.DataFrame([{c: 0.0 for c in cols}])

    df = bars_df.tail(lookback).copy()
    engine = _get_engine()
    features = engine.compute(df)

    # Take the last row and forward-fill NaNs → 0
    last = features.iloc[[-1]].copy()
    last = last.fillna(0.0)

    return last.reset_index(drop=True)
