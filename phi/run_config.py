"""
Phi-nance RunConfig + RunHistory
================================

Captures full backtest configuration for reproducibility.
Stored under: /runs/{run_id}/config.json, results.json, trades.csv
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_ROOT = _PROJECT_ROOT / "runs"


def _ensure_runs_dir() -> Path:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return _RUNS_ROOT


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class RunConfig:
    """
    Reproducible run configuration schema.
    """

    def __init__(
        self,
        dataset_id: str = "",
        symbols: List[str] = None,
        start_date: str = "",
        end_date: str = "",
        timeframe: str = "1D",
        vendor: str = "alphavantage",
        initial_capital: float = 100_000.0,
        trading_mode: str = "equities",  # equities | options
        indicators: Dict[str, Dict[str, Any]] = None,
        blend_method: str = "weighted_sum",
        blend_weights: Dict[str, float] = None,
        phiai_enabled: bool = False,
        phiai_constraints: Dict[str, Any] = None,
        exit_rules: Dict[str, Any] = None,
        position_sizing: Dict[str, Any] = None,
        evaluation_metric: str = "roi",
        **extra,
    ) -> None:
        self.dataset_id = dataset_id
        self.symbols = symbols or ["SPY"]
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.vendor = vendor
        self.initial_capital = initial_capital
        self.trading_mode = trading_mode
        self.indicators = indicators or {}
        self.blend_method = blend_method
        self.blend_weights = blend_weights or {}
        self.phiai_enabled = phiai_enabled
        self.phiai_constraints = phiai_constraints or {}
        self.exit_rules = exit_rules or {}
        self.position_sizing = position_sizing or {}
        self.evaluation_metric = evaluation_metric
        self.extra = extra

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timeframe": self.timeframe,
            "vendor": self.vendor,
            "initial_capital": self.initial_capital,
            "trading_mode": self.trading_mode,
            "indicators": self.indicators,
            "blend_method": self.blend_method,
            "blend_weights": self.blend_weights,
            "phiai_enabled": self.phiai_enabled,
            "phiai_constraints": self.phiai_constraints,
            "exit_rules": self.exit_rules,
            "position_sizing": self.position_sizing,
            "evaluation_metric": self.evaluation_metric,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        return cls(**d)


class RunHistory:
    """Manages run storage and retrieval."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _RUNS_ROOT

    def create_run(self, config: RunConfig) -> str:
        run_id = _new_run_id()
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        return run_id

    def save_results(self, run_id: str, results: Dict[str, Any], trades: Optional[pd.DataFrame] = None) -> None:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        if trades is not None and not trades.empty:
            trades.to_csv(path / "trades.csv", index=False)

    def load_config(self, run_id: str) -> Optional[RunConfig]:
        path = self.root / run_id / "config.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return RunConfig.from_dict(json.load(f))

    def load_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.root / run_id / "results.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        path = self.root / run_id / "trades.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with basic metadata."""
        runs = []
        if not self.root.exists():
            return runs
        for d in sorted(self.root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            cfg = self.load_config(d.name)
            res = self.load_results(d.name)
            runs.append({
                "run_id": d.name,
                "config": cfg.to_dict() if cfg else {},
                "results": res or {},
            })
        return runs[:100]  # last 100 runs
