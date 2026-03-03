"""
phinance.options.pricing
=========================

Black-Scholes European option pricing model.

Functions
---------
  black_scholes_call(S, K, T, r, sigma) → float
  black_scholes_put(S, K, T, r, sigma)  → float
  implied_volatility(option_price, S, K, T, r, option_type) → float

All inputs use standard quant conventions:
  S     — current underlying price
  K     — strike price
  T     — time to expiry in years
  r     — risk-free rate (annualised, continuous compounding)
  sigma — annualised implied/realised volatility
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ── Standard Normal CDF / PDF ─────────────────────────────────────────────────


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ── d1 / d2 ───────────────────────────────────────────────────────────────────


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ── Pricing ───────────────────────────────────────────────────────────────────


def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """European call option price under Black-Scholes.

    Parameters
    ----------
    S     : underlying spot price
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate (continuously compounded, e.g. 0.05)
    sigma : annualised volatility (e.g. 0.20)

    Returns
    -------
    float — call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """European put option price under Black-Scholes.

    Returns
    -------
    float — put option price
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ── Implied Volatility ────────────────────────────────────────────────────────


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Optional[float]:
    """Compute implied volatility via Newton-Raphson bisection.

    Parameters
    ----------
    option_price : float — market price of the option
    S, K, T, r   : float — standard BS inputs
    option_type  : ``"call"`` | ``"put"``
    tol          : float — convergence tolerance
    max_iter     : int   — maximum iterations

    Returns
    -------
    float — implied volatility, or ``None`` if it fails to converge
    """
    if T <= 0 or option_price <= 0:
        return None

    # Bisection between 1e-4 and 10.0
    lo, hi = 1e-4, 10.0
    fn = black_scholes_call if option_type == "call" else black_scholes_put

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = fn(S, K, T, r, mid)
        diff = price - option_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
