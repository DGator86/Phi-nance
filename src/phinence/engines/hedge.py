"""
Hedge Engine V1 — EOD dealer fields only.

EOD snapshot → GEX profile + derived phi + EPP proxy. No intraday vanna/charm
at hobbyist refresh rates. Prove daily signal first, then earn intraday upgrade.
"""

from __future__ import annotations

import math
from typing import Any

from phinence.contracts.assigned_packet import AssignedPacket


def black_scholes_vanilla(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float, float, float, float]:
    """Returns (delta, gamma, theta, vega, option_value). No div in V1."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    from math import log, sqrt, exp
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    # Call delta
    delta = exp(-q * T) * Nd1
    # Gamma (same for call/put)
    gamma = exp(-q * T) * (1 / sqrt(2 * math.pi) * math.exp(-0.5 * d1 * d1)) / (S * sigma * sqrt(T)) if S * sigma * sqrt(T) else 0
    # Vega (per 1% vol)
    vega = S * exp(-q * T) * (1 / sqrt(2 * math.pi) * math.exp(-0.5 * d1 * d1)) * sqrt(T) / 100
    # Theta (per day) simplified
    theta = 0.0  # stub
    opt_val = S * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2
    return delta, gamma, theta, vega, opt_val


def vanna_approx(S: float, K: float, T: float, sigma: float, delta: float, vega: float) -> float:
    """Vanna ≈ d(delta)/d(sigma). Approx: vega/S for near-ATM."""
    if S <= 0:
        return 0.0
    return float(vega / S)  # simplified


def charm_approx(delta: float, gamma: float, S: float, r: float, T: float) -> float:
    """Charm ≈ d(delta)/d(time). Simplified."""
    if T <= 0:
        return 0.0
    return float(-gamma * S * (r - 0.5 * gamma * S) / (2 * T))  # order-of-magnitude


def gex_from_chain(chain: dict[str, Any], spot: float, r: float = 0.05) -> dict[str, Any]:
    """
    EOD GEX/VEX profile from chain. Uses proshotv2 formulas:
    GEX = Gamma * OI * 100 * Spot^2 (calls +, puts -); VEX = Vanna * OI * 100 * Spot.
    """
    from phinence.engines.gex_math import aggregate_gex_vex
    out: dict[str, Any] = {
        "total_gex": 0.0,
        "total_vex": 0.0,
        "zero_gamma_strike": None,
        "walls": [],
        "epp_proxy": 0.0,
    }
    if spot <= 0:
        return out
    agg = aggregate_gex_vex(chain, spot, r=r, use_iv_from_chain=True)
    out["total_gex"] = agg["total_gex"]
    out["total_vex"] = agg["total_vex"]
    # EPP proxy: scalar from GEX level (calibrated later)
    out["epp_proxy"] = math.tanh(agg["total_gex"] / 1e9) if agg["total_gex"] else 0.0
    return out


class HedgeEngine:
    """EOD dealer fields only. V1: daily context/regime."""

    def __init__(self, dealer_field_frequency: str = "eod") -> None:
        assert dealer_field_frequency == "eod", "V1: dealer_field_frequency must be 'eod'"
        self.dealer_field_frequency = dealer_field_frequency

    def run(self, packet: AssignedPacket) -> dict[str, Any]:
        """Produce EOD dealer landmarks. No intraday steering in V1."""
        out: dict[str, Any] = {"gex_profile": {}, "epp_proxy": 0.0}
        if not packet.chain_snapshot or packet.chain_coverage.value == "missing":
            return out
        chain = packet.chain_snapshot
        # Spot: last close or quote (cablehead/proshot: quote for accuracy)
        spot = 0.0
        if packet.bars_1m:
            last = packet.bars_1m[-1]
            spot = float(last.get("close", 0) or 0)
        if spot <= 0 and chain.get("underlying"):
            spot = float(chain.get("underlying", {}).get("last", 0) or 0)
        if spot <= 0 and chain.get("raw", {}).get("underlying"):
            spot = float(chain["raw"].get("underlying", {}).get("last", 0) or 0)
        profile = gex_from_chain(chain, spot)
        out["gex_profile"] = profile
        out["epp_proxy"] = profile.get("epp_proxy", 0.0)
        return out
