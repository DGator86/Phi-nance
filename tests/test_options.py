from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from phi.options.backtest import run_options_backtest
from phi.options.contract import OptionContract, OptionType
from phi.options.position import OptionPosition
from phi.options.pricing import black_scholes_price, delta
from phi.run_config import RunConfig


def test_black_scholes_known_atm_call_price() -> None:
    px = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.CALL)
    assert px == pytest.approx(10.4506, rel=1e-3)


def test_black_scholes_known_atm_put_price() -> None:
    px = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.PUT)
    assert px == pytest.approx(5.5735, rel=1e-3)


def test_call_delta_bounds() -> None:
    d = delta(100.0, 100.0, 0.5, 0.02, 0.3, OptionType.CALL)
    assert 0.0 <= d <= 1.0


def test_option_contract_time_to_expiry() -> None:
    c = OptionContract("SPY", OptionType.CALL, 100.0, date(2024, 1, 31))
    assert c.time_to_expiry(date(2024, 1, 1)) == pytest.approx(30 / 365)


def test_option_position_mark_to_market_positive_for_itm_call() -> None:
    c = OptionContract("SPY", OptionType.CALL, 90.0, date(2024, 2, 1))
    p = OptionPosition(contract=c, quantity=1, entry_cost=0.0)
    mtm = p.mark_to_market(underlying_price=100.0, as_of=date(2024, 1, 15), r=0.02, sigma=0.2)
    assert mtm > 0


def test_run_config_requires_option_params_in_options_mode() -> None:
    with pytest.raises(ValueError):
        RunConfig(
            symbols=["SPY"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            trading_mode="options",
        )


def test_run_options_backtest_settles_on_expiry() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = [100, 102, 101, 103, 104, 106]
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1000] * 6}, index=idx)

    cfg = RunConfig(
        symbols=["SPY"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 6),
        trading_mode="options",
        initial_capital=100_000,
        option_params={
            "SPY": {
                "option_type": "call",
                "strike": 100.0,
                "expiry": date(2024, 1, 4),
                "iv": 0.3,
                "r": 0.02,
                "multiplier": 100,
            }
        },
    )

    result = run_options_backtest(cfg, df)
    actions = [t["action"] for t in result["trades"]]
    assert "buy" in actions
    assert "expiry" in actions
    assert isinstance(result["total_return"], float)
