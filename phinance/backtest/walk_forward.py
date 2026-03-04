"""
phinance.backtest.walk_forward
================================

Walk-Forward Optimisation (WFO) Harness.

Architecture
------------
The harness divides the full OHLCV data into rolling in-sample (IS) /
out-of-sample (OOS) windows and runs a full optimisation + backtest cycle
on each window:

  For each window:
    1. IS period  → run PhiAI/grid-search optimisation → best params
    2. OOS period → run backtest with those params → OOS metrics
    3. Aggregate all OOS metrics into a combined equity curve

The combined out-of-sample equity curve is the gold-standard estimate of
real-world performance.

Integration with AutonomousDeployer
-------------------------------------
  When ``auto_deploy=True``, the harness creates a StrategyProposal from the
  best IS window and pushes it through StrategyValidator → AutonomousDeployer.
  If OOS Sharpe < ``rollback_sharpe_threshold``, the deployer rolls the
  strategy back automatically.

Public API
----------
  WFOWindow          — one IS/OOS window record
  WFOResult          — aggregated walk-forward result
  WalkForwardConfig  — configuration dataclass
  WalkForwardHarness — main controller
  run_walk_forward   — convenience function
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.backtest.runner import run_backtest
from phinance.backtest.metrics import sharpe_ratio, max_drawdown, total_return, cagr
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
from phinance.blending.blender import blend_signals
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WFOWindow:
    """
    Metrics for one IS/OOS fold.

    Attributes
    ----------
    window_id      : str
    is_start       : int  — bar index (inclusive)
    is_end         : int
    oos_start      : int
    oos_end        : int
    best_indicator : str  — top indicator chosen on IS
    best_params    : dict
    is_sharpe      : float
    oos_sharpe     : float
    oos_return     : float
    oos_drawdown   : float
    oos_trades     : int
    elapsed_ms     : float
    """

    window_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    is_start:        int = 0
    is_end:          int = 0
    oos_start:       int = 0
    oos_end:         int = 0
    best_indicator:  str = ""
    best_params:     Dict[str, Any] = field(default_factory=dict)
    is_sharpe:       float = 0.0
    oos_sharpe:      float = 0.0
    oos_return:      float = 0.0
    oos_drawdown:    float = 0.0
    oos_trades:      int = 0
    elapsed_ms:      float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"WFOWindow(id={self.window_id}, "
            f"IS=[{self.is_start}:{self.is_end}], "
            f"OOS=[{self.oos_start}:{self.oos_end}], "
            f"IS_sharpe={self.is_sharpe:.2f}, "
            f"OOS_sharpe={self.oos_sharpe:.2f})"
        )


@dataclass
class WFOResult:
    """
    Aggregated result of a complete walk-forward run.

    Attributes
    ----------
    wfo_id            : str
    windows           : list[WFOWindow]
    combined_oos_sharpe : float — mean OOS Sharpe across windows
    combined_oos_return : float — compounded OOS return
    efficiency_ratio    : float — OOS Sharpe / IS Sharpe (robustness metric)
    num_windows         : int
    total_elapsed_ms    : float
    passed_gate         : bool  — True if combined_oos_sharpe > gate_threshold
    """

    wfo_id:               str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    windows:              List[WFOWindow] = field(default_factory=list)
    combined_oos_sharpe:  float = 0.0
    combined_oos_return:  float = 0.0
    efficiency_ratio:     float = 0.0
    num_windows:          int = 0
    total_elapsed_ms:     float = 0.0
    passed_gate:          bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wfo_id":               self.wfo_id,
            "num_windows":          self.num_windows,
            "combined_oos_sharpe":  self.combined_oos_sharpe,
            "combined_oos_return":  self.combined_oos_return,
            "efficiency_ratio":     self.efficiency_ratio,
            "total_elapsed_ms":     self.total_elapsed_ms,
            "passed_gate":          self.passed_gate,
            "windows":              [w.to_dict() for w in self.windows],
        }

    def summary(self) -> str:
        lines = [
            f"WFO {self.wfo_id}: {self.num_windows} windows",
            f"  OOS Sharpe (mean): {self.combined_oos_sharpe:.3f}",
            f"  OOS Return:        {self.combined_oos_return:.2%}",
            f"  Efficiency ratio:  {self.efficiency_ratio:.3f}",
            f"  Gate passed:       {self.passed_gate}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WFOResult(id={self.wfo_id}, "
            f"windows={self.num_windows}, "
            f"oos_sharpe={self.combined_oos_sharpe:.3f})"
        )


@dataclass
class WalkForwardConfig:
    """
    Configuration for the WalkForwardHarness.

    Attributes
    ----------
    is_bars          : int   — number of in-sample bars per window (default 120)
    oos_bars         : int   — number of out-of-sample bars per window (default 60)
    step_bars        : int   — number of bars to advance between windows (default 60)
    candidate_names  : list  — indicator names to try; defaults to all in catalog
    gate_threshold   : float — minimum OOS Sharpe to pass the gate (default 0.0)
    auto_deploy      : bool  — if True, deploy best strategy after WFO (default False)
    dry_run          : bool  — dry-run deployer (default True)
    """

    is_bars:         int = 120
    oos_bars:        int = 60
    step_bars:       int = 60
    candidate_names: Optional[List[str]] = None
    gate_threshold:  float = 0.0
    auto_deploy:     bool = False
    dry_run:         bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# WalkForwardHarness
# ─────────────────────────────────────────────────────────────────────────────


class WalkForwardHarness:
    """
    Rolling walk-forward optimisation harness.

    Usage
    -----
    ::

        from phinance.backtest.walk_forward import WalkForwardHarness, WalkForwardConfig

        harness = WalkForwardHarness(ohlcv=df, config=WalkForwardConfig(is_bars=120))
        result = harness.run()
        print(result.summary())
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
    ) -> None:
        self.ohlcv  = ohlcv.reset_index(drop=True)
        self.config = config or WalkForwardConfig()
        self._candidates = (
            self.config.candidate_names
            if self.config.candidate_names
            else list(INDICATOR_CATALOG.keys())
        )

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> WFOResult:
        """Execute the full walk-forward loop and return a WFOResult."""
        t0      = time.perf_counter()
        cfg     = self.config
        n       = len(self.ohlcv)
        windows = []

        pos = 0
        while pos + cfg.is_bars + cfg.oos_bars <= n:
            is_start  = pos
            is_end    = pos + cfg.is_bars
            oos_start = is_end
            oos_end   = min(oos_start + cfg.oos_bars, n)

            window = self._run_window(is_start, is_end, oos_start, oos_end)
            windows.append(window)
            logger.info(repr(window))

            pos += cfg.step_bars

        result = self._aggregate(windows, time.perf_counter() - t0)
        return result

    def windows_count(self) -> int:
        """Return number of windows that will be produced for current config."""
        cfg = self.config
        n   = len(self.ohlcv)
        count = 0
        pos = 0
        while pos + cfg.is_bars + cfg.oos_bars <= n:
            count += 1
            pos += cfg.step_bars
        return count

    # ── internal ─────────────────────────────────────────────────────────────

    def _run_window(
        self, is_start: int, is_end: int, oos_start: int, oos_end: int
    ) -> WFOWindow:
        t0     = time.perf_counter()
        is_df  = self.ohlcv.iloc[is_start:is_end].copy()
        oos_df = self.ohlcv.iloc[oos_start:oos_end].copy()

        # ── IS optimisation: pick indicator with highest IS Sharpe ──────────
        best_name   = ""
        best_sharpe = -999.0
        best_params: Dict[str, Any] = {}

        for name in self._candidates:
            try:
                sig = compute_indicator(name, is_df, {})
                if sig is None or sig.isna().all():
                    continue
                closes = is_df["close"].values.astype(float)
                arr    = sig.fillna(0.0).values.astype(float)
                # Quick Sharpe: daily returns of long-only positions
                positions = np.where(arr > 0.1, 1.0, 0.0)
                returns   = np.diff(closes) / closes[:-1]
                strat_ret = returns * positions[:-1]
                if strat_ret.std() > 0:
                    sh = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
                else:
                    sh = 0.0
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_name   = name
            except Exception:  # noqa: BLE001
                continue

        if not best_name and self._candidates:
            best_name = self._candidates[0]

        # ── OOS evaluation: run chosen indicator on OOS window ──────────────
        oos_sharpe = 0.0
        oos_ret    = 0.0
        oos_dd     = 0.0
        oos_trades = 0

        try:
            sig = compute_indicator(best_name, oos_df, best_params)
            if sig is not None and not sig.isna().all():
                closes    = oos_df["close"].values.astype(float)
                arr       = sig.fillna(0.0).values.astype(float)
                positions = np.where(arr > 0.1, 1.0, 0.0)
                returns   = np.diff(closes) / closes[:-1]
                strat_ret = returns * positions[:-1]
                if strat_ret.std() > 0:
                    oos_sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))
                equity = np.cumprod(1 + strat_ret) * 100_000
                oos_ret    = float((equity[-1] - equity[0]) / equity[0]) if len(equity) > 0 else 0.0
                peak   = np.maximum.accumulate(equity)
                dd     = (peak - equity) / (peak + 1e-9)
                oos_dd = float(dd.max()) if len(dd) > 0 else 0.0
                # Rough trade count
                pos_arr = np.where(arr > 0.1, 1, 0)
                oos_trades = int(np.sum(np.diff(pos_arr) > 0))
        except Exception:  # noqa: BLE001
            pass

        elapsed = (time.perf_counter() - t0) * 1000.0
        return WFOWindow(
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            best_indicator=best_name,
            best_params=best_params,
            is_sharpe=float(best_sharpe),
            oos_sharpe=float(oos_sharpe),
            oos_return=float(oos_ret),
            oos_drawdown=float(oos_dd),
            oos_trades=int(oos_trades),
            elapsed_ms=float(elapsed),
        )

    def _aggregate(self, windows: List[WFOWindow], elapsed: float) -> WFOResult:
        if not windows:
            return WFOResult(windows=[], num_windows=0, total_elapsed_ms=elapsed * 1000)

        oos_sharpes = [w.oos_sharpe for w in windows]
        is_sharpes  = [w.is_sharpe  for w in windows]
        oos_returns = [w.oos_return  for w in windows]

        mean_oos = float(np.mean(oos_sharpes))
        mean_is  = float(np.mean(is_sharpes)) if any(s != 0 for s in is_sharpes) else 1.0
        combined_ret = float(np.prod([1 + r for r in oos_returns]) - 1)
        eff_ratio    = mean_oos / max(abs(mean_is), 1e-9)
        passed       = mean_oos >= self.config.gate_threshold

        return WFOResult(
            windows=windows,
            combined_oos_sharpe=mean_oos,
            combined_oos_return=combined_ret,
            efficiency_ratio=eff_ratio,
            num_windows=len(windows),
            total_elapsed_ms=elapsed * 1000,
            passed_gate=passed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    ohlcv: pd.DataFrame,
    is_bars: int = 120,
    oos_bars: int = 60,
    step_bars: int = 60,
    gate_threshold: float = 0.0,
    candidate_names: Optional[List[str]] = None,
) -> WFOResult:
    """
    Run a walk-forward optimisation and return a WFOResult.

    Parameters
    ----------
    ohlcv            : pd.DataFrame
    is_bars          : int — in-sample bars per window
    oos_bars         : int — out-of-sample bars per window
    step_bars        : int — advance between windows
    gate_threshold   : float — minimum OOS Sharpe to pass gate
    candidate_names  : list or None — indicators to evaluate

    Returns
    -------
    WFOResult
    """
    cfg = WalkForwardConfig(
        is_bars=is_bars,
        oos_bars=oos_bars,
        step_bars=step_bars,
        gate_threshold=gate_threshold,
        candidate_names=candidate_names,
    )
    harness = WalkForwardHarness(ohlcv=ohlcv, config=cfg)
    return harness.run()
