from __future__ import annotations

from phi.options.models.binomial_tree import price_american, price_european
from phi.options.models.black_scholes import price_european as bs_price


def test_binomial_european_converges_to_black_scholes_call() -> None:
    b = price_european("call", 100, 100, 1.0, 0.05, 0.2, n=500)
    bs = bs_price("call", 100, 100, 1.0, 0.05, 0.2)
    assert abs(b - bs) < 0.05


def test_american_put_greater_than_european_put() -> None:
    american = price_american("put", 100, 100, 1.0, 0.01, 0.25, n=300)
    european = price_european("put", 100, 100, 1.0, 0.01, 0.25, n=300)
    assert american >= european
