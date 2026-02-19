"""
Scikit-learn Direction Classifier
-----------------------------------
Wraps three sklearn estimators (Random Forest, Gradient Boosting, Logistic
Regression) behind a common interface for next-bar direction prediction.

The classifier produces UP / DOWN predictions from regime features extracted
by `regime_engine.feature_extractor.get_regime_features()`.

Typical workflow:
    1. Train with `train_ml_classifier.py`
    2. Load in a strategy via `clf.load('models/classifier_rf.pkl')`
    3. Call `clf.predict(X)` or `clf.predict_proba(X)` each iteration
"""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


ModelType = Literal["random_forest", "gradient_boosting", "logistic"]


class DirectionClassifier:
    """
    Wraps sklearn classifiers for UP / DOWN next-bar direction prediction.

    Parameters
    ----------
    model_type : str
        One of 'random_forest', 'gradient_boosting', or 'logistic'.
    """

    def __init__(self, model_type: ModelType = "random_forest") -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn and joblib are required. "
                "Run: pip install scikit-learn joblib"
            )
        self.model_type = model_type
        self.scaler = StandardScaler()
        self._is_fitted = False

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
        elif model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose from: random_forest, gradient_boosting, logistic"
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_features: pd.DataFrame,
        y_labels: np.ndarray,
    ) -> "DirectionClassifier":
        """
        Train the classifier on historical regime features.

        Parameters
        ----------
        X_features : pd.DataFrame
            Feature matrix — each row is one sample (one trading day).
        y_labels : np.ndarray
            Integer labels: 1 = UP, 0 = DOWN.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_labels)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_features: pd.DataFrame) -> str:
        """
        Predict direction for the given feature row.

        Returns
        -------
        str
            'UP' or 'DOWN'.  Returns 'NEUTRAL' when no model is loaded.
        """
        if not self._is_fitted:
            return "NEUTRAL"
        X_scaled = self.scaler.transform(X_features)
        pred = self.model.predict(X_scaled)[0]
        return "UP" if pred == 1 else "DOWN"

    def predict_proba(self, X_features: pd.DataFrame) -> dict[str, float]:
        """
        Return class probabilities.

        Returns
        -------
        dict
            {'DOWN': p_down, 'UP': p_up}
            Both values are 0.5 when no model is loaded.
        """
        if not self._is_fitted:
            return {"DOWN": 0.5, "UP": 0.5}
        X_scaled = self.scaler.transform(X_features)
        probs = self.model.predict_proba(X_scaled)[0]
        return {"DOWN": float(probs[0]), "UP": float(probs[1])}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Persist model + scaler to disk."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"[DirectionClassifier] Saved to {path}")

    def load(self, path: str | pathlib.Path) -> "DirectionClassifier":
        """
        Load model + scaler from disk.

        Safe to call even when the file does not exist — the object remains
        unfitted and `.predict()` will return 'NEUTRAL'.
        """
        path = pathlib.Path(path)
        if not path.exists():
            print(
                f"[DirectionClassifier] No model file at {path}. "
                "Running unfitted — all predictions will be NEUTRAL. "
                "Train with train_ml_classifier.py first."
            )
            return self
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_fitted = True
        print(f"[DirectionClassifier] Loaded from {path}")
        return self
