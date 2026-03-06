"""Inference wrapper to expose transformer embeddings to agents."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from phinance.ml.data import FeatureScaler, prepare_market_features
from phinance.ml.transformer import MarketTransformer, MarketTransformerConfig


class TransformerFeatureExtractor:
    """Loads a saved transformer and returns cached embeddings from market windows."""

    def __init__(self, checkpoint_path: str | Path, cache_size: int = 512) -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
        config = MarketTransformerConfig(**payload["model_config"])

        self.model = MarketTransformer(config)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.feature_columns: list[str] = payload["feature_columns"]
        self.scaler = FeatureScaler.from_dict(payload["scaler"])
        self.sequence_length = int(payload["sequence_length"])
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _window_key(self, window_values: np.ndarray) -> int:
        return hash(window_values.tobytes())

    def extract_embedding(self, market_window: pd.DataFrame, pooling: str = "last") -> np.ndarray:
        engineered = prepare_market_features(market_window)
        if len(engineered) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows to build embedding")

        values = engineered[self.feature_columns].astype(float).to_numpy()[-self.sequence_length :]
        values = self.scaler.transform(values).astype(np.float32)
        key = self._window_key(values)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.embed(tensor, pooling=pooling).squeeze(0).cpu().numpy()

        self._cache[key] = emb
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return emb

    def extract_for_symbol(
        self,
        data_source_manager: Any,
        *,
        symbol: str,
        lookback: int | None = None,
        data_type: str = "bars",
        fetch_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Fetch history via ``DataSourceManager`` and return embedding."""
        fetch_kwargs = fetch_kwargs or {}
        rows = int(lookback or self.sequence_length)
        history = data_source_manager.fetch(data_type, cache_params={"symbol": symbol, "rows": rows}, symbol=symbol, rows=rows, **fetch_kwargs)
        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame(history)
        return self.extract_embedding(history)
