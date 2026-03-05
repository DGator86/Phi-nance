"""Unified Greeks API for options models."""

from __future__ import annotations

from typing import Dict

from .models import black_scholes, binomial_tree


def get_greeks(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = "bs",
    **kwargs,
) -> Dict[str, float]:
    """Return option Greeks for the selected pricing model."""
    model_key = model.lower().strip()
    if model_key in {"bs", "black_scholes", "black-scholes"}:
        return black_scholes.greeks(option_type, S, K, T, r, sigma)

    if model_key == "binomial":
        dS = float(kwargs.get("dS", max(S * 0.01, 1e-4)))
        d_sigma = float(kwargs.get("d_sigma", 1e-4))
        d_r = float(kwargs.get("d_r", 1e-4))
        d_t = float(kwargs.get("d_t", min(1 / 365, T / 10 if T > 0 else 1 / 365)))
        n = int(kwargs.get("n", 200))

        def p(s: float, k: float, t: float, rate: float, vol: float) -> float:
            return binomial_tree.price_american(option_type, s, k, max(t, 1e-8), rate, max(vol, 1e-8), n=n)

        p0 = p(S, K, T, r, sigma)
        p_up = p(S + dS, K, T, r, sigma)
        p_dn = p(max(S - dS, 1e-8), K, T, r, sigma)
        p_t_dn = p(S, K, max(T - d_t, 1e-8), r, sigma)

        return {
            "delta": (p_up - p_dn) / (2 * dS),
            "gamma": (p_up - 2 * p0 + p_dn) / (dS**2),
            "theta": (p_t_dn - p0) / d_t,
            "vega": (p(S, K, T, r, sigma + d_sigma) - p(S, K, T, r, max(sigma - d_sigma, 1e-8))) / (2 * d_sigma),
            "rho": (p(S, K, T, r + d_r, sigma) - p(S, K, T, r - d_r, sigma)) / (2 * d_r),
        }

    raise ValueError("model must be 'bs' or 'binomial'")
