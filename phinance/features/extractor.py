"""Feature extraction entrypoint for autoencoder and GP features."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phinance.features.autoencoder import MarketAutoencoder
from phinance.features.genetic_features import evaluate_expression
from phinance.features.registry import FeatureRegistry


class FeatureExtractor:
    """Loads discovered features and computes them on demand with caching."""

    def __init__(
        self,
        registry_path: str | Path,
        use_autoencoder: bool = True,
        use_gp_features: bool = True,
        window: int = 32,
    ) -> None:
        self.registry = FeatureRegistry.open(registry_path)
        self.use_autoencoder = use_autoencoder
        self.use_gp_features = use_gp_features
        self.window = window
        self._cache: dict[str, np.ndarray] = {}

        self.autoencoder: MarketAutoencoder | None = None
        auto_meta = self.registry.payload.get("autoencoder")
        if use_autoencoder and auto_meta and auto_meta.get("checkpoint_path"):
            self.autoencoder = MarketAutoencoder.load(auto_meta["checkpoint_path"])

    def _hash_window(self, frame: pd.DataFrame) -> str:
        values = frame.tail(self.window).select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        return hashlib.sha1(values.tobytes()).hexdigest()

    def extract(self, frame: pd.DataFrame) -> np.ndarray:
        cache_key = self._hash_window(frame)
        if cache_key in self._cache:
            return self._cache[cache_key]

        features: list[np.ndarray] = []
        numeric = frame.select_dtypes(include=[np.number])
        tail = numeric.tail(self.window)

        if self.use_autoencoder and self.autoencoder is not None and not tail.empty:
            window_vec = tail.to_numpy(dtype=np.float32).flatten()
            encoded = self.autoencoder.encode(window_vec)
            features.append(encoded.ravel())

        if self.use_gp_features:
            for entry in self.registry.top_gp_features(limit=16):
                expression = entry.get("expression")
                if not expression:
                    continue
                try:
                    val = evaluate_expression(expression, tail)
                except Exception:  # noqa: BLE001
                    val = 0.0
                features.append(np.array([val], dtype=np.float32))

        merged = np.concatenate(features).astype(np.float32) if features else np.array([], dtype=np.float32)
        self._cache[cache_key] = merged
        return merged

    @property
    def output_dim(self) -> int:
        auto_dim = 0
        if self.autoencoder is not None:
            auto_dim = self.autoencoder.config.latent_dim
        gp_dim = len(self.registry.payload.get("gp_features", [])) if self.use_gp_features else 0
        return auto_dim + gp_dim
