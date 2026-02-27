"""
Phi-nance Options Module
========================

Options backtest modes:
  - Simple delta simulation (original)
  - Full Black-Scholes engine with 9 strategy types (new)
"""

from .backtest import run_options_backtest
from .market_data import get_marketdataapp_snapshot
from .engine import (
    run_options_backtest_full,
    STRATEGY_NAMES,
    ENTRY_BIAS,
    black_scholes,
)

__all__ = [
    "run_options_backtest",
    "run_options_backtest_full",
    "get_marketdataapp_snapshot",
    "STRATEGY_NAMES",
    "ENTRY_BIAS",
    "black_scholes",
]
