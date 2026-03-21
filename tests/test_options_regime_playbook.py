"""Tests for options regime playbook and PnL attribution."""

from __future__ import annotations

import pandas as pd

from phi.options.backtest import _attribute_daily_pnl_by_regime, run_options_backtest
from phi.options.regime_playbook import (
    OptionsRegimePlaybook,
    PlaybookTransition,
    build_default_options_regime_playbook,
    playbook_entry_for_label,
    quick_detailed_regime_from_ohlcv,
)
from phi.run_config import RunConfig


def test_playbook_transition_from_json_keys() -> None:
    t = PlaybookTransition.model_validate(
        {
            "from": "A",
            "to": "B",
            "trigger": "t",
            "action": "act",
        }
    )
    assert t.from_regime == "A"
    assert t.to_regime == "B"


def test_default_playbook_covers_strategy_map() -> None:
    pb = build_default_options_regime_playbook()
    assert pb.version
    assert "BULL_LOW_VOL" in pb.regimes
    assert len(pb.transitions) >= 1


def test_playbook_entry_for_label() -> None:
    e = playbook_entry_for_label("BEAR_HIGH_VOL")
    assert e is not None
    assert "Long Put" in e.allowed_structures


def test_quick_detailed_regime_from_ohlcv() -> None:
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    close = pd.Series(range(100, 180), index=idx, dtype=float)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=idx,
    )
    lab = quick_detailed_regime_from_ohlcv(ohlcv)
    assert isinstance(lab, str)
    assert "_" in lab


def test_attribute_daily_pnl_by_regime() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    pv = [100.0, 101.0, 99.0, 102.0, 101.0]
    rs = pd.Series(["A", "A", "B", "B", "B"], index=idx)
    out = _attribute_daily_pnl_by_regime(pv, idx, rs)
    assert "A" in out
    assert "B" in out
    assert out["A"]["days"] == 1.0
    assert out["B"]["days"] == 3.0


def test_run_options_backtest_with_regime_series() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    close = pd.Series([100.0] * 20, index=idx)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1e6},
        index=idx,
    )
    rs = pd.Series(["state_0"] * len(idx), index=idx)
    sym = "TEST"
    cfg = RunConfig(
        symbols=[sym],
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        timeframe="1D",
        vendor="yfinance",
        initial_capital=1_000_000.0,
        trading_mode="options",
        indicators={"RSI": {"enabled": True, "params": {}}},
        blend_method="weighted_sum",
        blend_weights={"RSI": 1.0},
        option_params={
            sym: {
                "option_type": "call",
                "strike": 100.0,
                "expiry": idx[15].date(),
                "iv": 0.25,
                "r": 0.02,
                "quantity": 1,
            }
        },
    )
    res = run_options_backtest(cfg, ohlcv, regime_series=rs)
    assert "metrics_by_regime" in res
    assert res.get("regime_at_entry") == "state_0"
