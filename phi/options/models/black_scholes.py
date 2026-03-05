"""Black-Scholes pricing and Greeks for European options."""

from __future__ import annotations

import math
from typing import Dict

from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes ``d1`` term."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(max(S, 1e-12) / max(K, 1e-12)) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes ``d2`` term."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(max(T, 0.0))


def price_european(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call or put under Black-Scholes."""
    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if sigma <= 0:
        discounted_strike = K * math.exp(-r * T)
        return max(S - discounted_strike, 0.0) if option_type == "call" else max(discounted_strike - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def theta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    second_term = r * K * math.exp(-r * T)
    if option_type.lower() == "call":
        return first_term - second_term * norm.cdf(d2)
    return first_term + second_term * norm.cdf(-d2)


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)


def rho(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, sigma)
    disc = K * T * math.exp(-r * T)
    if option_type.lower() == "call":
        return disc * norm.cdf(d2)
    return -disc * norm.cdf(-d2)


def greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """Return a dictionary of Black-Scholes Greeks."""
    return {
        "delta": delta(option_type, S, K, T, r, sigma),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(option_type, S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "rho": rho(option_type, S, K, T, r, sigma),
    }


# Backward compatibility

def black_scholes_price(spot: float, strike: float, time_to_expiry: float, rate: float, volatility: float, option_type: str = "call") -> float:
    return price_european(option_type=option_type, S=spot, K=strike, T=time_to_expiry, r=rate, sigma=volatility)
