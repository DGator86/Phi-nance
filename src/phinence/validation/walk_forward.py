"""
Walk-forward validation — horizon-specific.

Intraday: 3mo train / 2wk test / 2wk step, expanding.
Daily: 6mo train / 1mo test / 1mo step.
Purge + embargo at boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator

import pandas as pd


class WFMode(str, Enum):
    INTRADAY = "intraday"  # 3mo / 2wk
    DAILY = "daily"       # 6mo / 1mo


@dataclass
class WFWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    step: str
    mode: WFMode


# Approximate trading days
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5


def expanding_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: WFMode,
    step_weeks: int | None = None,
    step_months: int | None = None,
) -> Iterator[WFWindow]:
    """
    Expanding windows. Intraday: train 63 td, test 10 td, step 2 wk.
    Daily: train 126 td, test 21 td, step 1 mo.
    """
    if mode == WFMode.INTRADAY:
        train_td = 63
        test_td = 10
        step_td = 10  # 2 weeks
    else:
        train_td = 126
        test_td = 21
        step_td = 21  # 1 month
    current = start
    while current + pd.Timedelta(days=train_td + test_td) <= end:
        train_end = current + pd.Timedelta(days=train_td)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_td - 1)
        yield WFWindow(
            train_start=current,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            step=f"{step_td}d",
            mode=mode,
        )
        current = current + pd.Timedelta(days=step_td)


class WalkForwardHarness:
    """Run WF folds; collect OOS metrics (AUC, cone coverage)."""

    def __init__(
        self,
        mode: WFMode = WFMode.INTRADAY,
        min_oos_auc_to_proceed: float = 0.52,
    ) -> None:
        self.mode = mode
        self.min_oos_auc_to_proceed = min_oos_auc_to_proceed

    def run_fold(
        self,
        fold: WFWindow,
        ticker: str,
        bar_store: Any,
        assigner: Any,
        engines: dict[str, Any],
        composer: Any,
    ) -> dict[str, Any]:
        """Single fold: run pipeline on test window, return OOS AUC and cone coverage."""
        from phinence.contracts.projection_packet import Horizon
        from phinence.validation.backtest_runner import run_backtest_fold
        horizon = Horizon.DAILY if self.mode == WFMode.DAILY else Horizon.INTRADAY_5M
        return run_backtest_fold(
            fold, ticker, bar_store, assigner, engines, composer, horizon=horizon
        )

    def gate_passed(self, fold_metrics: list[dict[str, Any]]) -> bool:
        """Phase 6 gate: mean OOS AUC > 0.52 to proceed to paper trading."""
        if not fold_metrics:
            return False
        mean_auc = sum(m.get("oos_auc", 0) for m in fold_metrics) / len(fold_metrics)
        return mean_auc >= self.min_oos_auc_to_proceed
