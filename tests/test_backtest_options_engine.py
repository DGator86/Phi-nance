from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from phi.backtest.options_engine import OptionsBacktestEngine


@dataclass
class _Leg:
    option_type: str = "call"
    strike: float = 100.0
    expiry: float = 0.2
    action: str = "buy"
    quantity: int = 1
    american: bool = False


class _Strategy:
    def __init__(self, legs):
        self._legs = legs

    def legs(self):
        return self._legs


def _ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=idx)


def test_options_engine_rejects_empty_data():
    engine = OptionsBacktestEngine()
    cfg = SimpleNamespace(initial_capital=100000.0, extra={"options_strategies": []})

    with pytest.raises(ValueError, match="data must contain"):
        engine.run(cfg, pd.DataFrame())


def test_options_engine_returns_no_trade_payload_without_positions():
    engine = OptionsBacktestEngine()
    cfg = SimpleNamespace(initial_capital=50_000.0, extra={"options_strategies": []})

    out = engine.run(cfg, _ohlcv())

    assert out["portfolio_value"] == [50_000.0]
    assert out["trade_log"] == []
    assert out["greeks"] == []


def test_options_engine_values_strategy_and_records_trade_log(monkeypatch):
    engine = OptionsBacktestEngine()
    cfg = SimpleNamespace(
        initial_capital=100_000.0,
        symbols=["SPY"],
        extra={
            "risk_free_rate": 0.01,
            "default_iv": 0.2,
            "options_strategies": [{"strategy": _Strategy([_Leg()]), "quantity": 1, "entry_index": 0}],
        },
    )

    # Keep model outputs deterministic for fast unit testing.
    monkeypatch.setattr("phi.backtest.options_engine.price_european", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        "phi.backtest.options_engine.greeks",
        lambda *_args, **_kwargs: {"delta": 0.5, "gamma": 0.1, "theta": -0.05, "vega": 0.2, "rho": 0.05},
    )

    out = engine.run(cfg, _ohlcv())

    assert len(out["portfolio_value"]) == 5
    assert out["total_return"] > 0
    assert len(out["trade_log"]) == 1
    assert out["trade_log"][0]["event"] == "leg_valued"
    assert len(out["greeks"]) == 5
