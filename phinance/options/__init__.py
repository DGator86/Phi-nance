"""
phinance.options — Options pricing, Greeks, IV surface, and backtesting.

Sub-modules
-----------
  pricing     — Black-Scholes option pricing (call, put, IV)
  greeks      — Delta, gamma, theta, vega, rho calculations
  iv_surface  — IV Surface: build, interpolate, smile, term-structure
  backtest    — Options-specific backtest (delta approximation)
  market_data — MarketDataApp connector for live chain snapshots
  ai_advisor  — AI-powered options strategy advisor
"""

from phinance.options.pricing import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility,
)
from phinance.options.greeks import compute_greeks, OptionsGreeks
from phinance.options.iv_surface import (
    IVSurface,
    IVPoint,
    build_iv_surface,
    interpolate_iv,
    smile_for_expiry,
    term_structure,
)
from phinance.options.backtest import run_options_backtest
from phinance.options.market_data import (
    OptionsSnapshot,
    MarketDataAppClient,
    get_marketdataapp_snapshot,
)

__all__ = [
    # pricing
    "black_scholes_call",
    "black_scholes_put",
    "implied_volatility",
    # greeks
    "compute_greeks",
    "OptionsGreeks",
    # iv_surface
    "IVSurface",
    "IVPoint",
    "build_iv_surface",
    "interpolate_iv",
    "smile_for_expiry",
    "term_structure",
    # backtest
    "run_options_backtest",
    # market_data
    "OptionsSnapshot",
    "MarketDataAppClient",
    "get_marketdataapp_snapshot",
]
