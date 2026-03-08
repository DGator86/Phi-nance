"""Hidden Markov Model detector for market regimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from phi.regime.base import RegimeDetector
from phi.regime.utils import extract_features


class HMMRegimeDetector(RegimeDetector):
    """Hidden Markov Model regime detector."""

    def __init__(self, n_states: int = 3, covariance_type: str = "diag", random_state: int = 42) -> None:
        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {
            "type": "hmm",
            "params": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
            },
            "features": [],
            "training_start": None,
            "training_end": None,
            "window": 20,
        }

    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> HMMRegimeDetector:
        """Fit HMM detector on extracted rolling features and return ``self``."""
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
                "features": self.feature_columns,
                "training_start": ohlcv.index.min(),
                "training_end": ohlcv.index.max(),
                "window": window,
            }
        )
        return self

    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Predict regime labels for each available bar."""
        if self.model is None:
            raise ValueError("HMMRegimeDetector must be fit before predict")

        window = int(self.metadata.get("window", 20))
        features = extract_features(ohlcv, window=window)
        features_filled = features.ffill().bfill().dropna()
        states = self.model.predict(features_filled.to_numpy())
        series = pd.Series([f"state_{int(s)}" for s in states], index=features_filled.index, name="regime")
        return series.reindex(ohlcv.index).ffill()

    def save(self, path: str | Path) -> None:
        """Persist trained HMM model and metadata to disk."""
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
        model_path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HMMRegimeDetector:
        """Load a persisted HMM detector from disk."""
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
