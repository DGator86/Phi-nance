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
    """KMeans/GMM detector that assigns regime cluster per bar."""

    def __init__(self, n_clusters: int = 3, method: Literal["kmeans", "gmm"] = "kmeans", random_state: int = 42) -> None:
        self.n_clusters = int(n_clusters)
        self.method = method
        self.random_state = random_state
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {
            "detector_class": self.__class__.__name__,
            "params": {
                "n_clusters": self.n_clusters,
                "method": self.method,
                "random_state": self.random_state,
            },
        }

    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> "ClusteringRegimeDetector":
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
        if self.model is None:
            raise ValueError("ClusteringRegimeDetector must be fit before predict")
        features = extract_features(ohlcv, window=int(self.metadata.get("window", 20)))
        x = features.to_numpy()
        labels = self.model.predict(x)
        return pd.Series([f"cluster_{int(c)}" for c in labels], index=features.index, name="regime")

    def save(self, path: str | Path) -> None:
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
        model_path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ClusteringRegimeDetector":
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
