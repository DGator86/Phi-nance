"""
phinance.options — Options pricing, Greeks, and backtesting.

Sub-modules
-----------
  pricing     — Black-Scholes option pricing
  greeks      — Delta, gamma, theta, vega calculations
  backtest    — Options-specific backtest (delta approximation)
  market_data — MarketDataApp connector for live chain snapshots
  ai_advisor  — AI-powered options strategy advisor
"""

from phinance.options.greeks import compute_greeks, OptionsGreeks
from phinance.options.backtest import run_options_backtest
from phinance.options.market_data import (
    OptionsSnapshot,
    MarketDataAppClient,
    get_marketdataapp_snapshot,
)

__all__ = [
    "compute_greeks",
    "OptionsGreeks",
    "run_options_backtest",
    "OptionsSnapshot",
    "MarketDataAppClient",
    "get_marketdataapp_snapshot",
]
