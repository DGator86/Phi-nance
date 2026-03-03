"""
phinance.storage.models
========================

Data model for stored backtest runs.

Classes
-------
  StoredRun — A fully-materialised run record: config + results + trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StoredRun:
    """A materialised backtest run record loaded from disk.

    Attributes
    ----------
    run_id   : str — unique run identifier (timestamp + hex suffix)
    config   : dict — serialised RunConfig
    results  : dict — backtest metrics (total_return, sharpe, ...)
    trades   : pd.DataFrame or None — closed trades table
    """

    run_id:  str
    config:  Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    trades:  Optional[pd.DataFrame] = None

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def symbols(self) -> List[str]:
        """Return the symbols from the stored config."""
        return self.config.get("symbols", [])

    @property
    def total_return(self) -> float:
        return float(self.results.get("total_return", 0.0))

    @property
    def sharpe(self) -> float:
        return float(self.results.get("sharpe", 0.0))

    @property
    def cagr(self) -> float:
        return float(self.results.get("cagr", 0.0))

    @property
    def max_drawdown(self) -> float:
        return float(self.results.get("max_drawdown", 0.0))

    def summary(self) -> str:
        """Return a one-line summary of this run."""
        sym = ", ".join(self.symbols) or "?"
        tf = self.config.get("timeframe", "?")
        start = self.config.get("start_date", "?")
        end = self.config.get("end_date", "?")
        return (
            f"[{self.run_id}] {sym} {tf} {start}→{end} | "
            f"ret={self.total_return:.1%} sharpe={self.sharpe:.2f} "
            f"dd={self.max_drawdown:.1%}"
        )

    def __repr__(self) -> str:
        return f"StoredRun(run_id={self.run_id!r}, symbols={self.symbols})"
