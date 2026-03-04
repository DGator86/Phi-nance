"""Tests for foundational phi.options pricing/greeks models."""

from __future__ import annotations

import math

from phi.options.models import Greeks, black_scholes_greeks, black_scholes_price


def test_black_scholes_call_put_parity():
    spot, strike, t, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.2
    call = black_scholes_price(spot, strike, t, rate, vol, "call")
    put = black_scholes_price(spot, strike, t, rate, vol, "put")
    lhs = call - put
    rhs = spot - strike * math.exp(-rate * t)
    assert abs(lhs - rhs) < 1e-8


def test_black_scholes_handles_expiry_intrinsic_value():
    assert black_scholes_price(110, 100, 0.0, 0.03, 0.2, "call") == 10.0
    assert black_scholes_price(90, 100, 0.0, 0.03, 0.2, "put") == 10.0


def test_greeks_output_type_and_signs_for_atm_call():
    greeks = black_scholes_greeks(100, 100, 1.0, 0.02, 0.2, "call")
    assert isinstance(greeks, Greeks)
    assert 0.0 < greeks.delta < 1.0
    assert greeks.gamma > 0
    assert greeks.vega > 0


def test_put_delta_and_rho_are_negative():
    greeks = black_scholes_greeks(100, 100, 1.0, 0.02, 0.2, "put")
    assert greeks.delta < 0
    assert greeks.rho < 0


def test_invalid_option_type_raises_value_error():
    try:
        black_scholes_price(100, 100, 1, 0.01, 0.2, "bad")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
