"""
phinance.backtest — Backtest engine and performance analytics.

Sub-modules
-----------
  models        — BacktestResult, Trade, Position data classes
  metrics       — Performance metric calculations (CAGR, Sharpe, drawdown, ...)
  engine        — Core vectorised simulation loop
  runner        — High-level orchestrator: accepts RunConfig → BacktestResult
  vectorized    — Ultra-fast NumPy-vectorized backtest engine
  walk_forward  — Walk-forward optimisation harness
  portfolio     — Multi-asset portfolio backtester

Public API
----------
    from phinance.backtest import run_backtest, BacktestResult
"""

from phinance.backtest.models import BacktestResult, Trade, Position
from phinance.backtest.runner import run_backtest
from phinance.backtest.vectorized import run_vectorized_backtest, VectorizedBacktestResult
from phinance.backtest.walk_forward import WalkForwardHarness, WalkForwardConfig, WFOResult, run_walk_forward
from phinance.backtest.portfolio import PortfolioBacktester, PortfolioConfig, PortfolioResult, run_portfolio_backtest
from phinance.backtest.distributed_runner import DistributedBacktestRunner

__all__ = [
    "BacktestResult", "Trade", "Position", "run_backtest",
    "run_vectorized_backtest", "VectorizedBacktestResult",
    "WalkForwardHarness", "WalkForwardConfig", "WFOResult", "run_walk_forward",
    "PortfolioBacktester", "PortfolioConfig", "PortfolioResult", "run_portfolio_backtest",
    "DistributedBacktestRunner",
]
