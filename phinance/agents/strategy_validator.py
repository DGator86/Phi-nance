"""
phinance.agents.strategy_validator
=====================================

StrategyValidator — runs a full walk-forward backtest on a proposed
strategy and decides whether it meets minimum performance thresholds
before live deployment.

Validation pipeline
-------------------
  1. Receives a ``StrategyProposal`` (from ``StrategyProposerAgent``).
  2. Runs ``run_backtest()`` with the proposed indicators and weights.
  3. Evaluates Sharpe ratio, max drawdown, and win rate against
     configurable minimum thresholds.
  4. Returns a ``ValidationResult`` with ``approved: bool``.

Public API
----------
  ValidationResult       — typed result dataclass
  StrategyValidator      — validates StrategyProposal objects
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from phinance.backtest.runner import run_backtest
from phinance.agents.strategy_proposer import StrategyProposal
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── ValidationResult ──────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Result of validating a StrategyProposal.

    Attributes
    ----------
    approved         : bool  — True if all thresholds are met
    proposal         : StrategyProposal
    sharpe           : float — annualised Sharpe ratio
    max_drawdown     : float — worst peak-to-trough drawdown (fraction)
    win_rate         : float — fraction of winning trades
    total_return     : float — fractional return over backtest period
    num_trades       : int   — number of completed trades
    rejection_reason : str   — human-readable reason if ``approved=False``
    backtest_stats   : dict  — full raw stats dict from run_backtest
    elapsed_ms       : float — validation wall-clock time
    """

    approved:         bool
    proposal:         StrategyProposal
    sharpe:           float = 0.0
    max_drawdown:     float = 0.0
    win_rate:         float = 0.0
    total_return:     float = 0.0
    num_trades:       int   = 0
    rejection_reason: str   = ""
    backtest_stats:   Dict[str, Any] = field(default_factory=dict)
    elapsed_ms:       float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved":         self.approved,
            "sharpe":           self.sharpe,
            "max_drawdown":     self.max_drawdown,
            "win_rate":         self.win_rate,
            "total_return":     self.total_return,
            "num_trades":       self.num_trades,
            "rejection_reason": self.rejection_reason,
            "elapsed_ms":       self.elapsed_ms,
        }


# ── StrategyValidator ─────────────────────────────────────────────────────────


class StrategyValidator:
    """
    Validates a StrategyProposal via walk-forward backtesting.

    Parameters
    ----------
    min_sharpe       : float — minimum acceptable Sharpe (default 0.3)
    max_drawdown     : float — maximum tolerable drawdown fraction (default 0.30)
    min_win_rate     : float — minimum win rate fraction (default 0.40)
    min_trades       : int   — minimum number of completed trades (default 2)
    initial_capital  : float — backtest starting capital (default 100_000)
    """

    def __init__(
        self,
        min_sharpe:      float = 0.3,
        max_drawdown:    float = 0.30,
        min_win_rate:    float = 0.40,
        min_trades:      int   = 2,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.min_sharpe      = min_sharpe
        self.max_drawdown    = max_drawdown
        self.min_win_rate    = min_win_rate
        self.min_trades      = min_trades
        self.initial_capital = initial_capital

    def validate(
        self,
        proposal: StrategyProposal,
        ohlcv:    pd.DataFrame,
        symbol:   str = "SIM",
    ) -> ValidationResult:
        """
        Run a full backtest on ``proposal`` and check thresholds.

        Parameters
        ----------
        proposal : StrategyProposal — the strategy to evaluate
        ohlcv    : pd.DataFrame     — historical OHLCV data
        symbol   : str              — ticker label for trade records

        Returns
        -------
        ValidationResult
        """
        t0 = time.time()

        try:
            result = run_backtest(
                ohlcv=ohlcv,
                symbol=symbol,
                indicators=proposal.indicators,
                blend_weights=proposal.weights,
                blend_method=proposal.blend_method,
                initial_capital=self.initial_capital,
            )
            stats = result.to_dict() if hasattr(result, "to_dict") else {}
        except Exception as exc:
            logger.warning("Backtest failed during validation: %s", exc)
            elapsed = (time.time() - t0) * 1000
            return ValidationResult(
                approved=False,
                proposal=proposal,
                rejection_reason=f"Backtest error: {exc}",
                elapsed_ms=elapsed,
            )

        sharpe       = float(stats.get("sharpe_ratio",  0.0) or 0.0)
        drawdown     = abs(float(stats.get("max_drawdown",  0.0) or 0.0))
        win_rate     = float(stats.get("win_rate",      0.0) or 0.0)
        total_return = float(stats.get("total_return",  0.0) or 0.0)
        num_trades   = int(stats.get("num_trades",      0)   or 0)

        # Evaluate thresholds
        reasons = []
        if sharpe < self.min_sharpe:
            reasons.append(f"Sharpe {sharpe:.3f} < {self.min_sharpe}")
        if drawdown > self.max_drawdown:
            reasons.append(f"Drawdown {drawdown:.3f} > {self.max_drawdown}")
        if win_rate < self.min_win_rate and num_trades >= self.min_trades:
            reasons.append(f"Win rate {win_rate:.3f} < {self.min_win_rate}")
        if num_trades < self.min_trades:
            reasons.append(f"Only {num_trades} trades (min {self.min_trades})")

        approved = len(reasons) == 0
        elapsed  = (time.time() - t0) * 1000

        vr = ValidationResult(
            approved=approved,
            proposal=proposal,
            sharpe=sharpe,
            max_drawdown=drawdown,
            win_rate=win_rate,
            total_return=total_return,
            num_trades=num_trades,
            rejection_reason="; ".join(reasons) if reasons else "",
            backtest_stats=stats,
            elapsed_ms=elapsed,
        )

        if approved:
            logger.info(
                "Strategy APPROVED | Sharpe=%.3f DD=%.3f WinRate=%.3f trades=%d",
                sharpe, drawdown, win_rate, num_trades,
            )
        else:
            logger.info("Strategy REJECTED: %s", vr.rejection_reason)

        return vr
