"""Hidden Markov Model detector for market regimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from phi.regime.base import RegimeDetector
from phi.regime.utils import extract_features


class HMMRegimeDetector(RegimeDetector):
    """Gaussian HMM-based regime detector."""

    def __init__(self, n_states: int = 3, covariance_type: str = "diag", random_state: int = 42) -> None:
        """Initialize detector hyperparameters and metadata scaffolding."""
        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {
            "detector_class": self.__class__.__name__,
            "params": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
            },
        }

    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> HMMRegimeDetector:
        """Fit Gaussian HMM parameters from engineered OHLCV features.

        Args:
            ohlcv: Historical bars to train on.
            **kwargs: Optional fit settings such as ``window`` and ``n_iter``.

        Returns:
            The fitted detector instance.
        """
        from hmmlearn.hmm import GaussianHMM

        window = int(kwargs.get("window", 20))
        features = extract_features(ohlcv, window=window)
        self.feature_columns = list(features.columns)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=int(kwargs.get("n_iter", 200)),
            random_state=self.random_state,
        )
        self.model.fit(features.to_numpy())

        self.metadata.update(
            {
                "feature_columns": self.feature_columns,
                "training_period": {
                    "start": str(features.index.min().date()),
                    "end": str(features.index.max().date()),
                },
                "window": window,
            }
        )
        return self

    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Predict discrete HMM state labels (``state_{i}``) for each eligible bar."""
        if self.model is None:
            raise ValueError("HMMRegimeDetector must be fit before predict")
        features = extract_features(ohlcv, window=int(self.metadata.get("window", 20)))
        states = self.model.predict(features.to_numpy())
        return pd.Series([f"state_{int(s)}" for s in states], index=features.index, name="regime")

    def save(self, path: str | Path) -> None:
        """Persist fitted model state and JSON metadata sidecar."""
        if self.model is None:
            raise ValueError("Cannot save an unfitted HMMRegimeDetector")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
                "feature_columns": self.feature_columns,
                "metadata": self.metadata,
            },
            model_path,
        )
        model_path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HMMRegimeDetector:
        """Load a previously saved HMM detector from disk."""
        payload = joblib.load(Path(path))
        inst = cls(
            n_states=int(payload["n_states"]),
            covariance_type=str(payload["covariance_type"]),
            random_state=int(payload["random_state"]),
        )
        inst.model = payload["model"]
        inst.feature_columns = list(payload.get("feature_columns", []))
        inst.metadata = dict(payload.get("metadata", {}))
        return inst
