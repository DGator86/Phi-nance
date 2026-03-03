"""
phinance.backtest — Backtest engine and performance analytics.

Sub-modules
-----------
  models  — BacktestResult, Trade, Position data classes
  metrics — Performance metric calculations (CAGR, Sharpe, drawdown, ...)
  engine  — Core vectorised simulation loop
  runner  — High-level orchestrator: accepts RunConfig → BacktestResult

Public API
----------
    from phinance.backtest import run_backtest, BacktestResult
"""

from phinance.backtest.models import BacktestResult, Trade, Position
from phinance.backtest.runner import run_backtest

__all__ = ["BacktestResult", "Trade", "Position", "run_backtest"]
