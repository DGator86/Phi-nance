"""Lumibot-compatible engine wrapper.

Currently delegates to vectorized engine when Lumibot runtime is unavailable,
so callers can use a stable interface while incremental migration continues.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from phi.backtest.engine import BacktestEngine
from phi.backtest.vectorized_engine import VectorizedEngine
from phi.run_config import RunConfig


class LumibotEngine(BacktestEngine):
    def run(self, config: RunConfig, data: pd.DataFrame) -> Dict[str, Any]:
        return VectorizedEngine().run(config, data)
