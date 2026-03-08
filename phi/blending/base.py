"""Base blending interface."""

from __future__ import annotations

from phi.logging import get_logger

logger = get_logger(__name__)

from abc import ABC, abstractmethod

import pandas as pd


class BlendMethod(ABC):
    """Abstract interface for all blend implementations."""

    @abstractmethod
    def blend(self, signals: pd.DataFrame, **kwargs) -> pd.Series:
        """Return composite signal series."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return tunable parameters for this method."""
