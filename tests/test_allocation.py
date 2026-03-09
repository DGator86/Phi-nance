from __future__ import annotations

from phi.backtest.allocation import (
    EqualWeightAllocation,
    FixedWeightAllocation,
    RiskParityAllocation,
    SignalWeightAllocation,
)


def test_equal_weight_allocation():
    a = EqualWeightAllocation()
    w = a.allocate(1000, {}, {"SPY": 100, "QQQ": 200})
    assert round(sum(w.values()), 8) == 1.0
    assert w["SPY"] == w["QQQ"]


def test_fixed_weight_allocation_normalizes():
    a = FixedWeightAllocation({"SPY": 0.7, "QQQ": 0.3})
    w = a.allocate(1000, {}, {"SPY": 100, "QQQ": 200})
    assert round(sum(w.values()), 8) == 1.0
    assert w["SPY"] > w["QQQ"]


def test_signal_weight_allocation():
    a = SignalWeightAllocation()
    w = a.allocate(1000, {"SPY": 2.0, "QQQ": 1.0}, {"SPY": 100, "QQQ": 100})
    assert round(sum(w.values()), 8) == 1.0
    assert w["SPY"] > w["QQQ"]


def test_risk_parity_allocation_inverse_volatility():
    a = RiskParityAllocation()
    w = a.allocate(1000, {}, {"SPY": 100, "QQQ": 100}, volatility={"SPY": 0.1, "QQQ": 0.2})
    assert round(sum(w.values()), 8) == 1.0
    assert w["SPY"] > w["QQQ"]
