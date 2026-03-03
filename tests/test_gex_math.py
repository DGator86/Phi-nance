"""GEX/Vanna math ported from proshotv2/gamma-vanna-options-exposure."""

import pytest

from phinence.engines.gex_math import (
    compute_vanna,
    gex_single_option,
    aggregate_gex_vex,
    norm_pdf,
    d1_d2,
)


def test_norm_pdf() -> None:
    assert 0 < norm_pdf(0) < 1
    assert norm_pdf(0) > norm_pdf(1)


def test_d1_d2() -> None:
    d1, d2 = d1_d2(100.0, 100.0, 0.05, 0.20, 0.25)
    assert d1 > 0
    assert d2 < d1


def test_compute_vanna() -> None:
    v = compute_vanna(100.0, 100.0, 0.05, 0.20, 0.25)
    assert v != 0


def test_gex_single_option_call() -> None:
    gex = gex_single_option(0.01, 1000, 100.0, "call")
    assert gex > 0


def test_gex_single_option_put() -> None:
    gex = gex_single_option(0.01, 1000, 100.0, "put")
    assert gex < 0


def test_aggregate_gex_vex_empty() -> None:
    out = aggregate_gex_vex({}, 100.0)
    assert out["total_gex"] == 0
    assert out["total_vex"] == 0


def test_aggregate_gex_vex_tradier_shape() -> None:
    # Symmetric call/put → net GEX 0
    chain_sym = {
        "options": {
            "option": [
                {"gamma": 0.02, "open_interest": 500, "option_type": "call", "strike": 100, "expiration_date": "2025-06-20"},
                {"gamma": 0.02, "open_interest": 500, "option_type": "put", "strike": 100, "expiration_date": "2025-06-20"},
            ]
        }
    }
    out = aggregate_gex_vex(chain_sym, 100.0)
    assert out["total_gex"] == 0  # call + put cancel
    # Calls only → positive GEX
    chain_calls = {"options": {"option": [{"gamma": 0.02, "open_interest": 1000, "option_type": "call", "strike": 100, "expiration_date": "2025-06-20"}]}}
    out2 = aggregate_gex_vex(chain_calls, 100.0)
    assert out2["total_gex"] > 0
