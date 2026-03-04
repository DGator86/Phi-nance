"""
phinance.strategies.ml_classifier
===================================

LightGBM classifier integrated as a first-class Phi-nance indicator.

``LGBMClassifierIndicator`` trains a LightGBM binary classifier on the
feature matrix produced by ``phinance.strategies.ml_features`` and emits
a continuous signal in [−1, +1] equal to ``2 × P(up) − 1``.

Training strategy
-----------------
A rolling walk-forward re-fit is performed inside ``compute()``:
  1. Features are built from the full OHLCV series.
  2. The first ``train_size`` bars are used to fit the model.
  3. The model predicts the remaining bars (out-of-sample).
  4. Every ``retrain_every`` bars the model is re-fit on an expanding window.

This avoids look-ahead bias while keeping the model current.

Persistence
-----------
Trained models are optionally saved to / loaded from disk via ``joblib``.
Set ``model_path`` to enable persistence:
  ``ind = LGBMClassifierIndicator(model_path="models/lgbm_spy.pkl")``

References
----------
* Ke et al. (2017) — "LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree" (NeurIPS 2017)
* Prado (2018) — "Advances in Financial Machine Learning"

Public API
----------
  LGBMClassifierIndicator  — BaseIndicator subclass, fully catalog-compatible
  train_lgbm_model(...)    — standalone training function for pre-fitting
  load_lgbm_model(path)    — load a saved model
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator
from phinance.strategies.ml_features import build_features, build_labels

try:
    import lightgbm as lgb
    import joblib
    LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LGBM_AVAILABLE = False

from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_LGBM_PARAMS: dict = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     200,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           1,
    "verbose":          -1,
}


# ── Standalone helpers ────────────────────────────────────────────────────────


def train_lgbm_model(
    ohlcv: pd.DataFrame,
    horizon: int = 1,
    label_threshold: float = 0.0,
    lgbm_params: Optional[dict] = None,
    model_path: Optional[str] = None,
) -> "lgb.LGBMClassifier":
    """Train a LightGBM classifier on OHLCV data.

    Parameters
    ----------
    ohlcv           : pd.DataFrame — OHLCV data
    horizon         : int          — forward-return label horizon (bars)
    label_threshold : float        — minimum return to label as 1
    lgbm_params     : dict, optional — override default LightGBM params
    model_path      : str, optional  — save trained model to this path

    Returns
    -------
    lgb.LGBMClassifier (fitted)

    Raises
    ------
    ImportError — if lightgbm/joblib not installed
    """
    if not LGBM_AVAILABLE:
        raise ImportError("pip install lightgbm joblib")

    params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
    X = build_features(ohlcv)
    y = build_labels(ohlcv, horizon=horizon, threshold=label_threshold)

    valid = y.notna()
    X_train = X[valid]
    y_train = y[valid].astype(int)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model, model_path)
        logger.info("LGBM model saved → %s", model_path)

    return model


def load_lgbm_model(path: str) -> "lgb.LGBMClassifier":
    """Load a previously saved LGBMClassifier from disk.

    Parameters
    ----------
    path : str — path to joblib file

    Returns
    -------
    lgb.LGBMClassifier
    """
    if not LGBM_AVAILABLE:
        raise ImportError("pip install lightgbm joblib")
    return joblib.load(path)


# ── Indicator class ───────────────────────────────────────────────────────────


class LGBMClassifierIndicator(BaseIndicator):
    """LightGBM binary classifier as a Phi-nance indicator.

    Signal = ``2 × P(up) − 1`` → range [−1, +1].

    Positive signal → model predicts upward move (buy).
    Negative signal → model predicts downward move (sell).

    Parameters
    ----------
    train_size     : int   — bars used for initial training (default 252)
    retrain_every  : int   — re-fit every N bars on expanding window (default 63)
    horizon        : int   — forward-return label horizon (default 1)
    label_threshold: float — min return magnitude for up-label (default 0.0)
    model_path     : str   — optional path to load/save model
    lgbm_params    : dict  — LightGBM hyperparameters
    """

    name = "LGBM Classifier"
    default_params = {
        "train_size":      252,
        "retrain_every":   63,
        "horizon":         1,
        "label_threshold": 0.0,
        "model_path":      None,
    }
    param_grid: dict = {}   # PhiAI grid: left empty (tuned via lgbm_params)

    def __init__(
        self,
        lgbm_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._lgbm_params = lgbm_params
        self._model: Optional["lgb.LGBMClassifier"] = None

    def compute(
        self,
        df: pd.DataFrame,
        train_size: int = 252,
        retrain_every: int = 63,
        horizon: int = 1,
        label_threshold: float = 0.0,
        model_path: Optional[str] = None,
        **_: Any,
    ) -> pd.Series:
        """Compute LightGBM probability signal.

        Parameters
        ----------
        df             : pd.DataFrame — OHLCV
        train_size     : int          — initial training window
        retrain_every  : int          — expanding re-fit cadence (bars)
        horizon        : int          — forward-label horizon
        label_threshold: float        — return threshold for up-label
        model_path     : str, optional — load pre-trained model instead of fitting

        Returns
        -------
        pd.Series in [−1, +1]
        """
        if not LGBM_AVAILABLE:
            logger.warning("lightgbm not installed — returning zero signal")
            return pd.Series(0.0, index=df.index, name=self.name)

        n = len(df)
        if n < train_size + horizon + 5:
            logger.warning(
                "LGBM: insufficient data (%d rows, need %d) — zero signal",
                n, train_size + horizon + 5,
            )
            return pd.Series(0.0, index=df.index, name=self.name)

        X_full = build_features(df)
        signal  = np.zeros(n)

        # Try loading a pre-trained model first
        if model_path and os.path.exists(model_path):
            try:
                model = load_lgbm_model(model_path)
                proba = model.predict_proba(X_full)[:, 1]
                signal = 2.0 * proba - 1.0
                return pd.Series(signal, index=df.index, name=self.name).clip(-1, 1)
            except Exception as exc:
                logger.warning("Failed to load LGBM model from %s: %s", model_path, exc)

        # Walk-forward training
        params = {**_DEFAULT_LGBM_PARAMS, **(self._lgbm_params or {})}
        y_full = build_labels(df, horizon=horizon, threshold=label_threshold)

        train_end = train_size
        while train_end <= n - horizon:
            # Fit on [0, train_end)
            X_train = X_full.iloc[:train_end]
            y_train = y_full.iloc[:train_end]
            valid   = y_train.notna()
            if valid.sum() < 20:
                train_end += retrain_every
                continue

            try:
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train[valid], y_train[valid].astype(int))
                self._model = model
            except Exception as exc:
                logger.warning("LGBM fit failed at bar %d: %s", train_end, exc)
                train_end += retrain_every
                continue

            # Predict on the next ``retrain_every`` bars (or until end)
            pred_end = min(train_end + retrain_every, n)
            X_pred   = X_full.iloc[train_end:pred_end]
            if len(X_pred) == 0:
                break
            try:
                proba = model.predict_proba(X_pred)[:, 1]
                signal[train_end:pred_end] = 2.0 * proba - 1.0
            except Exception as exc:
                logger.warning("LGBM predict failed: %s", exc)

            train_end = pred_end

        result = pd.Series(signal, index=df.index, name=self.name)
        return result.clip(-1.0, 1.0).fillna(0.0)
