"""
LightGBM Direction Classifier
---------------------------------
Production-grade gradient-boosting classifier for UP/DOWN direction
prediction on regime features.

Advantages over sklearn GBM:
  • Much faster training on large datasets (histogram-based splits)
  • Native support for missing values
  • Built-in early stopping w/ validation set
  • Feature importance via split count OR information gain

Typical workflow:
    1. Train with `train_ml_classifier.py --model lightgbm`
    2. Load in a strategy: `clf.load('models/classifier_lgb.txt')`
    3. Call `clf.predict(X)` or `clf.predict_proba(X)` each iteration
"""

from __future__ import annotations

import pathlib
from typing import List

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


class LightGBMDirectionClassifier:
    """
    Binary LightGBM classifier for next-bar direction prediction.

    Parameters
    ----------
    num_leaves : int
        Tree complexity parameter (higher = more expressive, more overfit risk).
    learning_rate : float
        Boosting step size.
    num_rounds : int
        Maximum number of boosting rounds (early stopping may cut short).
    """

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        num_rounds: int = 300,
    ) -> None:
        if not _LGB_AVAILABLE:
            raise ImportError(
                "lightgbm is required. Run: pip install lightgbm"
            )
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }
        self.num_rounds = num_rounds
        self.model: lgb.Booster | None = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_features: pd.DataFrame,
        y_labels: np.ndarray,
        validation_split: float = 0.2,
    ) -> "LightGBMDirectionClassifier":
        """
        Train with early stopping on a hold-out validation slice.

        Parameters
        ----------
        X_features : pd.DataFrame
            Feature matrix (one row per trading day, columns = feature names).
        y_labels : np.ndarray
            Binary labels — 1 = UP, 0 = DOWN.
        validation_split : float
            Fraction of the tail to use as validation (time-series safe).
        """
        n_train = int(len(X_features) * (1 - validation_split))
        X_train, y_train = X_features.iloc[:n_train], y_labels[:n_train]
        X_val,   y_val   = X_features.iloc[n_train:], y_labels[n_train:]

        self.feature_names = X_features.columns.tolist()

        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            reference=train_data,
        )

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),   # silent
            ],
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_features: pd.DataFrame) -> str:
        """
        Predict direction.

        Returns
        -------
        str
            'UP' or 'DOWN'.  'NEUTRAL' if no model is loaded.
        """
        if self.model is None:
            return "NEUTRAL"
        proba = self.model.predict(X_features)[0]
        return "UP" if proba > 0.5 else "DOWN"

    def predict_proba(self, X_features: pd.DataFrame) -> dict[str, float]:
        """
        Return class probabilities.

        Returns
        -------
        dict
            {'DOWN': p_down, 'UP': p_up}
        """
        if self.model is None:
            return {"DOWN": 0.5, "UP": 0.5}
        proba = float(self.model.predict(X_features)[0])
        return {"DOWN": 1.0 - proba, "UP": proba}

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Return a DataFrame of the top-N most important features by gain.

        Parameters
        ----------
        top_n : int
            Number of features to return.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded yet.")
        importance = self.model.feature_importance(importance_type="gain")
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Persist model to disk (LightGBM native text format)."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model not trained yet.")
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        print(f"[LightGBMDirectionClassifier] Saved to {path}")

    def load(self, path: str | pathlib.Path) -> "LightGBMDirectionClassifier":
        """
        Load model from disk.

        Safe to call when the file does not exist — model stays None and
        `.predict()` returns 'NEUTRAL'.
        """
        path = pathlib.Path(path)
        if not path.exists():
            print(
                f"[LightGBMDirectionClassifier] No model file at {path}. "
                "Running without model — all predictions will be NEUTRAL. "
                "Train with train_ml_classifier.py first."
            )
            return self
        self.model = lgb.Booster(model_file=str(path))
        self.feature_names = self.model.feature_name()
        print(f"[LightGBMDirectionClassifier] Loaded from {path}")
        return self
