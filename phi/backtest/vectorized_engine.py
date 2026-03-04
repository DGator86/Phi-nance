"""Simple vectorized backtest engine."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from phi.backtest.direct import run_direct_backtest
from phi.backtest.engine import BacktestEngine
from phi.run_config import RunConfig


class VectorizedEngine(BacktestEngine):
    def run(self, config: RunConfig, data: pd.DataFrame) -> Dict[str, Any]:
        symbol = (config.symbols or ["SPY"])[0]
        results, strat = run_direct_backtest(
            ohlcv=data,
            symbol=symbol,
            indicators=config.indicators,
            blend_weights=config.blend_weights,
            blend_method=config.blend_method,
            signal_threshold=float(config.extra.get("signal_threshold", 0.15)),
            initial_capital=float(config.initial_capital),
        )
        return {"metrics": results, "trades": getattr(strat, "prediction_log", [])}
