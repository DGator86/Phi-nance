from __future__ import annotations

import pandas as pd

from phi.backtest.direct import run_portfolio_backtest


def _mk(start: str, p0: float) -> pd.DataFrame:
    idx = pd.date_range(start, periods=30, freq="D")
    close = pd.Series([p0 + i for i in range(30)], index=idx)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1}, index=idx)


def test_run_portfolio_backtest_multi_asset_runs():
    data = {"SPY": _mk("2024-01-01", 100), "QQQ": _mk("2024-01-01", 200)}
    res = run_portfolio_backtest(
        data_dict=data,
        indicators={"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        blend_weights={"RSI": 1.0},
        allocation_strategy="equal_weight",
        rebalance_frequency="W",
    )
    assert "portfolio_value" in res
    assert len(res["portfolio_value"]) > 0
    assert abs(sum(res["final_weights"].values()) - 1.0) < 1e-6
