from __future__ import annotations

import math

from phi.options.models.black_scholes import greeks, price_european


def test_black_scholes_reference_call_value() -> None:
    price = price_european("call", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert abs(price - 10.4506) < 1e-3


def test_black_scholes_reference_put_value() -> None:
    price = price_european("put", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert abs(price - 5.5735) < 1e-3


def test_put_call_parity_holds() -> None:
    c = price_european("call", 120, 100, 0.5, 0.03, 0.25)
    p = price_european("put", 120, 100, 0.5, 0.03, 0.25)
    assert abs((c - p) - (120 - 100 * math.exp(-0.03 * 0.5))) < 1e-8


def test_greeks_signs_for_call() -> None:
    g = greeks("call", 100, 100, 1, 0.01, 0.2)
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0
    assert g["vega"] > 0
