"""
phinance.options.greeks
========================

Black-Scholes option Greeks calculations.

Classes / Functions
-------------------
  OptionsGreeks       — Dataclass holding all Greeks
  compute_greeks(...)  — Compute all Greeks from BS inputs

Greeks
------
  delta  — dV/dS   — sensitivity to underlying price change
  gamma  — d²V/dS² — rate of change of delta
  theta  — dV/dt   — daily time decay
  vega   — dV/dσ   — sensitivity to 1% change in volatility
  rho    — dV/dr   — sensitivity to interest rate change
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from phinance.options.pricing import _d1, _d2, _norm_cdf, _norm_pdf


@dataclass
class OptionsGreeks:
    """Container for all standard option Greeks.

    Attributes
    ----------
    delta : float — rate of change of option price w.r.t. underlying
    gamma : float — rate of change of delta w.r.t. underlying
    theta : float — daily dollar time decay (negative for long options)
    vega  : float — price change per 1% move in IV
    rho   : float — price change per 1% move in risk-free rate
    """

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega":  self.vega,
            "rho":   self.rho,
        }


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> OptionsGreeks:
    """Compute all standard Black-Scholes Greeks.

    Parameters
    ----------
    S           : underlying spot price
    K           : strike price
    T           : time to expiry in years
    r           : risk-free rate (continuous, e.g. 0.05)
    sigma       : annualised volatility (e.g. 0.20)
    option_type : ``"call"`` | ``"put"``

    Returns
    -------
    OptionsGreeks
    """
    if T <= 0 or sigma <= 0:
        # At expiry / degenerate
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return OptionsGreeks(
            delta=1.0 if (option_type == "call" and S > K) else 0.0,
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        )

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    df = math.exp(-r * T)

    # ── Delta ─────────────────────────────────────────────────────────────────
    if option_type == "call":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0

    # ── Gamma (same for call and put) ─────────────────────────────────────────
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)

    # ── Theta (annualised → convert to daily / 365) ───────────────────────────
    term1 = -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == "call":
        term2 = r * K * df * _norm_cdf(d2)
        theta_annual = term1 - term2
    else:
        term2 = r * K * df * _norm_cdf(-d2)
        theta_annual = term1 + term2
    theta = theta_annual / 365  # daily theta

    # ── Vega (per 1% change in sigma) ────────────────────────────────────────
    vega = S * _norm_pdf(d1) * sqrt_T * 0.01

    # ── Rho (per 1% change in r) ──────────────────────────────────────────────
    if option_type == "call":
        rho = K * T * df * _norm_cdf(d2) * 0.01
    else:
        rho = -K * T * df * _norm_cdf(-d2) * 0.01

    return OptionsGreeks(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
    )
