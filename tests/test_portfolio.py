from __future__ import annotations

import pandas as pd

from phi.backtest.portfolio import Order, Portfolio


def test_portfolio_buy_sell_and_value():
    p = Portfolio(initial_capital=1000)
    p.update_prices({"SPY": 100.0})
    p.execute_order(Order(symbol="SPY", shares=5, price=100.0))

    assert p.cash == 500.0
    assert p.positions["SPY"] == 5
    assert p.total_value() == 1000.0

    p.update_prices({"SPY": 110.0})
    assert p.total_value() == 1050.0

    p.execute_order(Order(symbol="SPY", shares=-5, price=110.0))
    assert p.positions.get("SPY") is None
    assert p.cash == 1050.0


def test_portfolio_returns_series():
    p = Portfolio(initial_capital=1000)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    p.equity_curve = [(idx[0], 1000.0), (idx[1], 1010.0), (idx[2], 1005.0)]
    r = p.returns()
    assert len(r) == 3
    assert r.iloc[0] == 0.0
