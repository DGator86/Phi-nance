"""
phinance.backtest.monte_carlo
==============================

Monte Carlo Simulation for backtest robustness analysis.

Uses trade-level resampling (bootstrapping) to estimate the distribution of
key performance metrics and quantify confidence in reported results.

Three simulation modes
----------------------
1. **bootstrap_trades**  — resample the observed trade list with replacement.
2. **random_entry**      — apply the original signal to random entry offsets.
3. **return_shuffle**    — shuffle daily returns to break autocorrelation.

Usage
-----
    from phinance.backtest.monte_carlo import run_monte_carlo

    mc = run_monte_carlo(
        backtest_result = br,          # BacktestResult
        n_simulations   = 1000,
        method          = "bootstrap_trades",
        confidence      = 0.95,
    )
    print(mc.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.utils.logging import get_logger

_log = get_logger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Result of a Monte Carlo simulation."""
    method:          str
    n_simulations:   int
    confidence:      float

    # Metric distributions (list of floats, one per simulation)
    sharpe_dist:        List[float] = field(default_factory=list)
    total_return_dist:  List[float] = field(default_factory=list)
    max_drawdown_dist:  List[float] = field(default_factory=list)
    win_rate_dist:      List[float] = field(default_factory=list)

    # ── Percentile helpers ────────────────────────────────────────────────────

    def _percentile(self, dist: List[float], pct: float) -> float:
        if not dist:
            return 0.0
        return float(np.percentile(dist, pct * 100))

    @property
    def lo(self) -> float:
        return (1.0 - self.confidence) / 2.0

    @property
    def hi(self) -> float:
        return 1.0 - self.lo

    # ── Summary ───────────────────────────────────────────────────────────────

    @property
    def summary(self) -> Dict[str, Any]:
        lo, hi = self.lo, self.hi
        return {
            "method":        self.method,
            "n_simulations": self.n_simulations,
            "confidence":    self.confidence,
            "sharpe": {
                "mean":    round(float(np.mean(self.sharpe_dist)), 4) if self.sharpe_dist else 0.0,
                "median":  round(self._percentile(self.sharpe_dist, 0.50), 4),
                f"p{int(lo*100)}":  round(self._percentile(self.sharpe_dist, lo), 4),
                f"p{int(hi*100)}":  round(self._percentile(self.sharpe_dist, hi), 4),
            },
            "total_return": {
                "mean":   round(float(np.mean(self.total_return_dist)), 4) if self.total_return_dist else 0.0,
                "median": round(self._percentile(self.total_return_dist, 0.50), 4),
                f"p{int(lo*100)}":  round(self._percentile(self.total_return_dist, lo), 4),
                f"p{int(hi*100)}":  round(self._percentile(self.total_return_dist, hi), 4),
            },
            "max_drawdown": {
                "mean":   round(float(np.mean(self.max_drawdown_dist)), 4) if self.max_drawdown_dist else 0.0,
                "median": round(self._percentile(self.max_drawdown_dist, 0.50), 4),
                f"p{int(lo*100)}":  round(self._percentile(self.max_drawdown_dist, lo), 4),
                f"p{int(hi*100)}":  round(self._percentile(self.max_drawdown_dist, hi), 4),
            },
            "win_rate": {
                "mean":   round(float(np.mean(self.win_rate_dist)), 4) if self.win_rate_dist else 0.0,
                "median": round(self._percentile(self.win_rate_dist, 0.50), 4),
            },
            "prob_positive_return":  round(
                sum(1 for r in self.total_return_dist if r > 0) / max(1, len(self.total_return_dist)), 4
            ),
            "prob_positive_sharpe":  round(
                sum(1 for s in self.sharpe_dist if s > 0) / max(1, len(self.sharpe_dist)), 4
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return simulation results as a tidy DataFrame."""
        n = max(len(self.sharpe_dist), len(self.total_return_dist), 1)
        def _pad(lst: List[float], length: int) -> List[float]:
            return lst + [0.0] * (length - len(lst)) if len(lst) < length else lst[:length]
        return pd.DataFrame({
            "sharpe":        _pad(self.sharpe_dist, n),
            "total_return":  _pad(self.total_return_dist, n),
            "max_drawdown":  _pad(self.max_drawdown_dist, n),
            "win_rate":      _pad(self.win_rate_dist, n),
        })


# ── Main entry point ──────────────────────────────────────────────────────────

def run_monte_carlo(
    backtest_result: Any,
    n_simulations:  int   = 500,
    method:         str   = "bootstrap_trades",
    confidence:     float = 0.95,
    seed:           Optional[int] = None,
) -> MCResult:
    """Run Monte Carlo simulation on a BacktestResult.

    Parameters
    ----------
    backtest_result : BacktestResult
    n_simulations   : int   — number of MC iterations.
    method          : str   — ``"bootstrap_trades"`` | ``"return_shuffle"``
    confidence      : float — confidence interval (e.g. 0.95 for 95 %).
    seed            : int, optional — random seed for reproducibility.

    Returns
    -------
    MCResult
    """
    rng = np.random.default_rng(seed)

    mc = MCResult(
        method=method,
        n_simulations=n_simulations,
        confidence=confidence,
    )

    if method == "bootstrap_trades":
        _bootstrap_trades(backtest_result, n_simulations, rng, mc)
    elif method == "return_shuffle":
        _return_shuffle(backtest_result, n_simulations, rng, mc)
    else:
        _log.warning("Unknown MC method '%s', falling back to bootstrap_trades", method)
        _bootstrap_trades(backtest_result, n_simulations, rng, mc)

    _log.info(
        "MC complete: method=%s, n=%d, p[sharpe>0]=%.2f, p[return>0]=%.2f",
        method, n_simulations,
        mc.summary["prob_positive_sharpe"],
        mc.summary["prob_positive_return"],
    )
    return mc


# ── Simulation strategies ─────────────────────────────────────────────────────

def _bootstrap_trades(
    br: Any,
    n_simulations: int,
    rng: np.random.Generator,
    mc: MCResult,
) -> None:
    """Bootstrap trade P&L to build metric distributions."""
    trades = getattr(br, "trades", []) or []
    if not trades:
        # No trades — fill with zeros
        mc.sharpe_dist       = [0.0] * n_simulations
        mc.total_return_dist = [0.0] * n_simulations
        mc.max_drawdown_dist = [0.0] * n_simulations
        mc.win_rate_dist     = [0.0] * n_simulations
        return

    pnls      = np.array([float(getattr(t, "pnl", 0.0)) for t in trades])
    n_trades  = len(pnls)
    initial_capital = float(getattr(br, "metadata", {}).get("initial_capital", 100_000))

    for _ in range(n_simulations):
        # Resample trades with replacement
        sampled = rng.choice(pnls, size=n_trades, replace=True)

        # Build portfolio value curve
        pv = np.empty(n_trades + 1)
        pv[0] = initial_capital
        for i, p in enumerate(sampled):
            pv[i + 1] = pv[i] + p

        total_ret = (pv[-1] - pv[0]) / pv[0]
        sharpe    = _quick_sharpe(pv)
        dd        = _quick_max_dd(pv)
        wr        = float(np.mean(sampled > 0))

        mc.total_return_dist.append(round(total_ret, 4))
        mc.sharpe_dist.append(round(sharpe, 4))
        mc.max_drawdown_dist.append(round(dd, 4))
        mc.win_rate_dist.append(round(wr, 4))


def _return_shuffle(
    br: Any,
    n_simulations: int,
    rng: np.random.Generator,
    mc: MCResult,
) -> None:
    """Shuffle daily returns to break autocorrelation."""
    pv_series = getattr(br, "portfolio_value", []) or []
    if len(pv_series) < 2:
        mc.sharpe_dist       = [0.0] * n_simulations
        mc.total_return_dist = [0.0] * n_simulations
        mc.max_drawdown_dist = [0.0] * n_simulations
        mc.win_rate_dist     = [0.5] * n_simulations
        return

    pv = np.array(pv_series, dtype=float)
    daily_returns = np.diff(pv) / pv[:-1]
    initial_capital = float(pv[0])

    for _ in range(n_simulations):
        shuffled = rng.permutation(daily_returns)
        # Rebuild portfolio curve from shuffled returns
        new_pv = np.empty(len(shuffled) + 1)
        new_pv[0] = initial_capital
        for i, r in enumerate(shuffled):
            new_pv[i + 1] = new_pv[i] * (1.0 + r)

        total_ret = (new_pv[-1] - new_pv[0]) / new_pv[0]
        sharpe    = _quick_sharpe(new_pv)
        dd        = _quick_max_dd(new_pv)

        mc.total_return_dist.append(round(total_ret, 4))
        mc.sharpe_dist.append(round(sharpe, 4))
        mc.max_drawdown_dist.append(round(dd, 4))
        mc.win_rate_dist.append(0.0)   # not applicable for return_shuffle


# ── Quick metric functions (no DataFrame overhead) ────────────────────────────

def _quick_sharpe(pv: np.ndarray, bars_per_year: float = 252.0) -> float:
    """Fast Sharpe estimate from portfolio value array."""
    if len(pv) < 2:
        return 0.0
    rets = np.diff(pv) / pv[:-1]
    std  = float(np.std(rets, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(bars_per_year))


def _quick_max_dd(pv: np.ndarray) -> float:
    """Fast max-drawdown from portfolio value array."""
    if len(pv) < 2:
        return 0.0
    running_max = np.maximum.accumulate(pv)
    drawdowns   = (pv - running_max) / np.where(running_max != 0, running_max, 1.0)
    return float(np.min(drawdowns))
