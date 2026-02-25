"""
Phi-nance Options Module
========================

Options backtest mode:
  - Long Call/Put (delta-based simulation)
  - Exit: profit %, stop %, time-based
"""

from .backtest import run_options_backtest

__all__ = ["run_options_backtest"]
