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



def run_portfolio_backtest_engine(config: RunConfig, data: dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compatibility wrapper for the portfolio-aware direct engine."""
    from phi.backtest.direct import run_portfolio_backtest

    return run_portfolio_backtest(
        data_dict=data,
        indicators=config.indicators,
        blend_weights=config.blend_weights,
        blend_method=config.blend_method,
        initial_capital=config.initial_capital,
        allocation_strategy=getattr(config, "allocation_strategy", "equal_weight"),
        allocation_params=getattr(config, "allocation_params", {}),
        rebalance_frequency=getattr(config, "rebalance_frequency", "M"),
        rebalance_threshold=getattr(config, "rebalance_threshold", None),
    )
