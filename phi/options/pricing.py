"""Black-Scholes pricing and Greeks for European options."""

from __future__ import annotations

import math

from .contract import OptionType


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def _intrinsic(S: float, K: float, option_type: OptionType) -> float:
    if option_type == OptionType.CALL:
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Return Black-Scholes price for a European call or put."""
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if T <= 0:
        return _intrinsic(S, K, option_type)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    disc_k = K * math.exp(-r * T)
    if option_type == OptionType.CALL:
        return S * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
    return disc_k * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
    if T <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma)
    if option_type == OptionType.CALL:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    first = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    disc = K * math.exp(-r * T)
    if option_type == OptionType.CALL:
        return first - r * disc * _norm_cdf(d2)
    return first + r * disc * _norm_cdf(-d2)
