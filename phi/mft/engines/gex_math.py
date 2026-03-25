"""
GEX / Vanna math ported from proshotv2/gamma-vanna-options-exposure.

References:
- GEX = Gamma * OI * 100 * Spot^2 (perfiliev.co.uk)
- Vanna = exp(-q*T) * norm.pdf(d1) * (d2 / sigma); VEX = Vanna * OI * 100 * Spot
"""

from __future__ import annotations

import math
from typing import Any


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    """Black-Scholes d1, d2."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def compute_vanna(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """
    Vanna = d(delta)/d(sigma). Ported from proshotv2:
    vanna = exp(-q*T) * norm.pdf(d1) * (d2 / sigma)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * norm_pdf(d1) * (d2 / sigma)


def gex_single_option(
    gamma: float,
    open_interest: int,
    spot: float,
    option_type: str,
    *,
    multiplier: float = 100.0,
) -> float:
    """
    Gamma exposure for one option. GEX = Gamma * OI * 100 * Spot^2.
    Calls: positive GEX (dealer long gamma); Puts: negative (dealer short).
    """
    if spot <= 0:
        return 0.0
    gex = gamma * open_interest * multiplier * (spot * spot)
    if (option_type or "").lower() in ("put", "p"):
        return -gex
    return gex


def vex_single_option(
    vanna: float,
    open_interest: int,
    spot: float,
    *,
    multiplier: float = 100.0,
) -> float:
    """Vanna exposure: VEX = Vanna * OI * 100 * Spot."""
    return vanna * open_interest * multiplier * spot


def _option_list(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Tradier/cablehead chain to list of option dicts."""
    options = chain.get("options") or chain
    if isinstance(options, list):
        return options
    opt = options.get("option")
    if opt is None:
        # Nested call/put
        out = (options.get("call") or []) + (options.get("put") or [])
        return out if isinstance(out, list) else [out] if out else []
    return opt if isinstance(opt, list) else [opt] if opt else []


def aggregate_gex_vex(
    chain: dict[str, Any],
    spot: float,
    r: float = 0.05,
    *,
    use_iv_from_chain: bool = True,
) -> dict[str, Any]:
    """
    Total GEX and VEX from chain. Options must have gamma, open_interest; for VEX
    need IV or we approximate from delta. Spot required.
    """
    total_gex = 0.0
    total_vex = 0.0
    opts = _option_list(chain)
    for o in opts:
        gamma = float(o.get("gamma") or 0)
        oi = int(o.get("open_interest") or 0)
        opt_type = (o.get("option_type") or o.get("type") or "call").lower()
        total_gex += gex_single_option(gamma, oi, spot, opt_type)

        if use_iv_from_chain and (o.get("greeks") or o.get("iv")):
            greeks = o.get("greeks") or {}
            iv = float(greeks.get("mid_iv") or greeks.get("smv_vol") or o.get("iv") or 0)
            strike = o.get("strike") or greeks.get("strike")
            if iv > 0 and strike is not None and o.get("expiration_date"):
                try:
                    from datetime import datetime, date
                    exp = o.get("expiration_date")
                    if isinstance(exp, str):
                        exp_dt = datetime.fromisoformat(exp.replace("Z", "").split("T")[0])
                    else:
                        exp_dt = exp
                    if hasattr(exp_dt, "date"):
                        exp_date = exp_dt.date()
                    else:
                        exp_date = exp_dt
                    today = date.today()
                    T = max(1 / 365.0, (exp_date - today).days / 365.25)
                    vanna = compute_vanna(spot, float(strike), r, iv, T, 0.0)
                    total_vex += vex_single_option(vanna, oi, spot)
                except Exception:
                    pass
    return {"total_gex": total_gex, "total_vex": total_vex}
