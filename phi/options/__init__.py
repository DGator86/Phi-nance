"""
Phi-nance Options Module
========================

Options backtest mode:
  - Long Call/Put (delta-based simulation)
  - Exit: profit %, stop %, time-based
"""

from .backtest import run_options_backtest, compute_greeks
from .market_data import get_marketdataapp_snapshot

__all__ = ["run_options_backtest", "compute_greeks", "get_marketdataapp_snapshot"]
