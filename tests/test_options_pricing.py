from __future__ import annotations

import pytest

from phi.options.backtest import compute_greeks
from phi.options.contract import OptionType
from phi.options.pricing import black_scholes_price


def test_black_scholes_price_matches_known_reference_values():
    # Known reference near S=100 K=100 T=1 r=5% sigma=20%
    call = black_scholes_price(100, 100, 1.0, 0.05, 0.2, OptionType.CALL)
    put = black_scholes_price(100, 100, 1.0, 0.05, 0.2, OptionType.PUT)

    assert call == pytest.approx(10.4506, rel=1e-3)
    assert put == pytest.approx(5.5735, rel=1e-3)


def test_black_scholes_handles_zero_time_to_expiry():
    assert black_scholes_price(110, 100, 0.0, 0.02, 0.3, OptionType.CALL) == 10.0
    assert black_scholes_price(90, 100, 0.0, 0.02, 0.3, OptionType.PUT) == 10.0


def test_black_scholes_rejects_zero_volatility():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        black_scholes_price(100, 100, 1.0, 0.01, 0.0, OptionType.CALL)


def test_compute_greeks_returns_expected_shape_and_signs():
    greeks = compute_greeks(100, 100, 0.5, 0.02, 0.25, OptionType.CALL)

    assert set(greeks) == {"delta", "gamma", "theta", "vega"}
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
