"""Base interfaces for trainable market regime detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class RegimeDetector(ABC):
    """Abstract contract for regime detector implementations.

    Implementations are expected to maintain a ``metadata`` dictionary containing,
    at minimum:
      - ``detector_class`` (str): concrete class name.
      - ``params`` (dict[str, Any]): constructor/training parameters.
    Additional keys such as ``feature_columns``, ``training_period``, and ``window``
    are recommended when available.
    """

    metadata: dict[str, Any]

    @abstractmethod
    def fit(self, ohlcv: pd.DataFrame, **kwargs: Any) -> RegimeDetector:
        """Train detector on historical OHLCV data and return ``self``.

        Args:
            ohlcv: Price data containing OHLCV columns.
            **kwargs: Model-specific training options.

        Returns:
            The fitted detector instance.
        """

    @abstractmethod
    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Predict per-bar regime labels indexed by bar timestamp."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist trained detector and metadata to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> RegimeDetector:
        """Load a persisted detector instance from disk."""

    @classmethod
    def get_param_grid(cls) -> dict[str, list[Any]]:
        """Optional coarse hyperparameter grid for simple search flows."""
        return {}
