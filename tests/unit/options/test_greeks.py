from __future__ import annotations

from phi.options.greeks import get_greeks


def test_get_greeks_black_scholes() -> None:
    g = get_greeks("call", 100, 100, 1.0, 0.02, 0.2, model="bs")
    assert set(g) == {"delta", "gamma", "theta", "vega", "rho"}


def test_get_greeks_binomial_finite_differences() -> None:
    g = get_greeks("put", 100, 95, 0.5, 0.02, 0.3, model="binomial", n=200)
    assert set(g) == {"delta", "gamma", "theta", "vega", "rho"}
    assert g["gamma"] >= 0
