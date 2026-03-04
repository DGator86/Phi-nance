"""Closed-form Greeks for European options under Black-Scholes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .black_scholes import _d1, _d2


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> Greeks:
    """Compute Black-Scholes Greeks for a call or put."""
    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        call_delta = 1.0 if spot > strike else 0.0
        delta = call_delta if option_type == "call" else call_delta - 1.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = _d1(spot, strike, time_to_expiry, rate, volatility)
    d2 = _d2(spot, strike, time_to_expiry, rate, volatility)
    pdf_d1 = _norm_pdf(d1)
    nd1 = _norm_cdf(d1)
    discount = math.exp(-rate * time_to_expiry)

    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = nd1
        theta = (
            -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t)
            - rate * strike * discount * _norm_cdf(d2)
        )
        rho = strike * time_to_expiry * discount * _norm_cdf(d2)
    else:
        delta = nd1 - 1.0
        theta = (
            -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t)
            + rate * strike * discount * _norm_cdf(-d2)
        )
        rho = -strike * time_to_expiry * discount * _norm_cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
