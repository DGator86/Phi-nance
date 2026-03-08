"""Backtest engine interface."""

from __future__ import annotations

from phi.logging import get_logger

logger = get_logger(__name__)

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from phi.run_config import RunConfig


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, config: RunConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute a backtest and return results (metrics + artifacts)."""
