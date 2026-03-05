"""Dataclass adapter for Black-Scholes Greeks."""

from __future__ import annotations

from dataclasses import dataclass

from . import black_scholes


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
    vals = black_scholes.greeks(option_type, spot, strike, time_to_expiry, rate, volatility)
    return Greeks(**vals)
