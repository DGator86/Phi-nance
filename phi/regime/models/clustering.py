"""Clustering-based regime detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from phi.regime.base import RegimeDetector
from phi.regime.utils import extract_features


class ClusteringRegimeDetector(RegimeDetector):
    """KMeans/GMM detector that assigns a regime cluster to each bar."""

    def __init__(self, n_clusters: int = 3, method: Literal["kmeans", "gmm"] = "kmeans", random_state: int = 42) -> None:
        """Initialize clustering detector and metadata container."""
        self.n_clusters = int(n_clusters)
        self.method = method
        self.random_state = random_state
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {
            "type": self.method,
            "params": {
                "n_clusters": self.n_clusters,
                "method": self.method,
                "random_state": self.random_state,
            },
            "features": [],
            "training_start": None,
            "training_end": None,
            "window": 20,
        }

    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> ClusteringRegimeDetector:
        """Fit clustering detector on extracted features and return ``self``."""
        window = int(kwargs.get("window", 20))
        features = extract_features(ohlcv, window=window)
        x = features.to_numpy()

        if self.method == "gmm":
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
            self.model.fit(x)
        else:
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
            self.model.fit(x)

        self.feature_columns = list(features.columns)
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
        """Predict cluster labels for each available bar."""
        if self.model is None:
            raise ValueError("ClusteringRegimeDetector must be fit before predict")

        features = extract_features(ohlcv, window=int(self.metadata.get("window", 20)))
        labels = self.model.predict(features.to_numpy())
        series = pd.Series([f"cluster_{int(c)}" for c in labels], index=features.index, name="regime")
        return series.reindex(ohlcv.index).ffill()

    def save(self, path: str | Path) -> None:
        """Persist trained clustering model and metadata to disk."""
        if self.model is None:
            raise ValueError("Cannot save an unfitted ClusteringRegimeDetector")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_clusters": self.n_clusters,
                "method": self.method,
                "random_state": self.random_state,
                "feature_columns": self.feature_columns,
                "metadata": self.metadata,
            },
            model_path,
        )
        model_path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ClusteringRegimeDetector:
        """Load a persisted clustering detector from disk."""
        payload = joblib.load(Path(path))
        inst = cls(
            n_clusters=int(payload["n_clusters"]),
            method=str(payload["method"]),
            random_state=int(payload["random_state"]),
        )
        inst.model = payload["model"]
        inst.feature_columns = list(payload.get("feature_columns", []))
        inst.metadata = dict(payload.get("metadata", {}))
        return inst
