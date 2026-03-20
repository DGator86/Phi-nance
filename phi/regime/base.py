"""Base interfaces for trainable market regime detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class RegimeDetector(ABC):
    """Abstract base class for regime detectors.

    Subclasses must implement ``fit``, ``predict``, ``save``, and ``load``.
    Implementations should maintain a ``metadata`` dictionary containing at least:
    - ``type``: detector key (for example ``hmm`` or ``kmeans``)
    - ``params``: training/model parameters
    - ``features``: feature column names used during training
    - ``training_start`` and ``training_end``: training period boundaries
    """

    metadata: dict[str, Any]

    @abstractmethod
    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> RegimeDetector:
        """Train detector on historical OHLCV data and return ``self``."""

    @abstractmethod
    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Predict per-bar regime labels indexed by bar timestamp."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist trained detector and metadata to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> RegimeDetector:
        """Load detector instance from disk."""
