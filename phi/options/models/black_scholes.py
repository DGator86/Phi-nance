"""Black-Scholes pricing utilities for European options."""

from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(spot: float, strike: float, time_to_expiry: float, rate: float, volatility: float) -> float:
    """Compute d1 term for Black-Scholes equations."""
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    return (
        math.log(max(spot, 1e-12) / max(strike, 1e-12))
        + (rate + 0.5 * volatility * volatility) * time_to_expiry
    ) / sigma_sqrt_t


def _d2(spot: float, strike: float, time_to_expiry: float, rate: float, volatility: float) -> float:
    """Compute d2 term for Black-Scholes equations."""
    return _d1(spot, strike, time_to_expiry, rate, volatility) - volatility * math.sqrt(max(time_to_expiry, 0.0))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Price a European call/put using Black-Scholes."""
    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if time_to_expiry <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)

    if volatility <= 0:
        forward_intrinsic = max(spot - strike * math.exp(-rate * time_to_expiry), 0.0)
        if option_type == "call":
            return float(forward_intrinsic)
        return float(forward_intrinsic - spot + strike * math.exp(-rate * time_to_expiry))

    d1 = _d1(spot, strike, time_to_expiry, rate, volatility)
    d2 = _d2(spot, strike, time_to_expiry, rate, volatility)
    discount = math.exp(-rate * time_to_expiry)

    if option_type == "call":
        return float(spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2))

    return float(strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1))
