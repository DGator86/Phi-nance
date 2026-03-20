from __future__ import annotations

import pandas as pd

from phi.backtest.options_engine import OptionsBacktestEngine
from phi.options.strategies import VerticalSpread
from phi.run_config import RunConfig


def test_options_backtest_runs_vertical_spread():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    close = [100 + i * 0.2 for i in range(50)]
    data = pd.DataFrame({"close": close}, index=idx)
    chain = pd.DataFrame({"strike": [95, 100, 105, 95, 100, 105], "expiry": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3], "iv": [0.2, 0.2, 0.21, 0.22, 0.22, 0.23]})

    cfg = RunConfig(initial_capital=100_000, trading_mode="options", options_strategies=[{"strategy": VerticalSpread("call", 95, 105, 0.25, 1)}], iv_chain_data=chain)
    result = OptionsBacktestEngine().run(cfg, data)

    assert "portfolio_value" in result
    assert len(result["portfolio_value"]) == len(data)
    assert "greeks" in result and len(result["greeks"]) == len(data)
