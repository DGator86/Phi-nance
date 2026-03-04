"""
phi.backtest.run_history — Run History Manager
================================================
Each run is stored under:

  /runs/{run_id}/
    config.json
    results.json
    trades.csv

RunHistory provides list / load / compare / delete operations.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .run_config import RunConfig

RUNS_ROOT = Path(__file__).parents[2] / "runs"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_run(
    config: RunConfig,
    results: Dict[str, Any],
    trades: Optional[pd.DataFrame] = None,
) -> Path:
    """Persist a completed run.  Returns the run directory path."""
    run_dir = _run_dir(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    with open(run_dir / "config.json", "w") as f:
        f.write(config.to_json())

    # results.json — strip non-serializable objects
    safe_results = _make_serializable(results)
    with open(run_dir / "results.json", "w") as f:
        json.dump(safe_results, f, indent=2)

    # trades.csv
    if trades is not None and not trades.empty:
        trades.to_csv(run_dir / "trades.csv", index=False)

    return run_dir


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a run by ID.  Returns dict with keys: config, results, trades."""
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        return None

    result: Dict[str, Any] = {"run_id": run_id}

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            result["config"] = RunConfig.from_json(cfg_path.read_text())
        except Exception:
            result["config"] = None

    res_path = run_dir / "results.json"
    if res_path.exists():
        try:
            result["results"] = json.loads(res_path.read_text())
        except Exception:
            result["results"] = {}

    trades_path = run_dir / "trades.csv"
    if trades_path.exists():
        try:
            result["trades"] = pd.read_csv(trades_path)
        except Exception:
            result["trades"] = pd.DataFrame()

    return result


def delete_run(run_id: str) -> bool:
    """Remove a run directory.  Returns True if existed."""
    run_dir = _run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# List / Query
# ─────────────────────────────────────────────────────────────────────────────

def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return summary dicts for all stored runs, newest first.

    Each dict contains: run_id, created_at, symbol, timeframe,
    trading_mode, indicators, blend_mode, primary_metric, metric_value,
    initial_capital, end_capital, net_pnl_pct.
    """
    if not RUNS_ROOT.exists():
        return []

    summaries: List[Dict[str, Any]] = []

    for run_dir in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue

        summary: Dict[str, Any] = {"run_id": run_dir.name}

        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            try:
                d = json.loads(cfg_path.read_text())
                summary.update({
                    "created_at":     d.get("created_at", "")[:19],
                    "symbol":         d.get("symbol", ""),
                    "timeframe":      d.get("timeframe", ""),
                    "trading_mode":   d.get("trading_mode", ""),
                    "blend_mode":     d.get("blend_mode", ""),
                    "initial_capital": d.get("initial_capital", 0),
                    "eval_metric":    d.get("evaluation_metric", ""),
                    "indicators":     [i.get("display_name", i.get("name", ""))
                                       for i in d.get("indicators", [])
                                       if i.get("enabled", True)],
                    "phiai":          d.get("phiai_enabled", False),
                })
            except Exception:
                pass

        res_path = run_dir / "results.json"
        if res_path.exists():
            try:
                r = json.loads(res_path.read_text())
                metrics = r.get("metrics", {})
                summary.update({
                    "total_return":   metrics.get("total_return",   None),
                    "cagr":           metrics.get("cagr",           None),
                    "sharpe":         metrics.get("sharpe",         None),
                    "max_drawdown":   metrics.get("max_drawdown",   None),
                    "profit_factor":  metrics.get("profit_factor",  None),
                    "win_rate":       metrics.get("win_rate",       None),
                    "n_trades":       metrics.get("n_trades",       0),
                    "end_capital":    r.get("end_capital",          None),
                })
            except Exception:
                pass

        summaries.append(summary)
        if len(summaries) >= limit:
            break

    return summaries


def compare_runs(run_ids: List[str]) -> pd.DataFrame:
    """Load multiple runs and return a comparison DataFrame."""
    rows = []
    for rid in run_ids:
        run = load_run(rid)
        if run is None:
            continue
        cfg = run.get("config")
        res = run.get("results", {})
        metrics = res.get("metrics", {})
        row = {
            "run_id":        rid,
            "symbol":        cfg.symbol if cfg else "",
            "timeframe":     cfg.timeframe if cfg else "",
            "indicators":    ", ".join(i.display_name for i in (cfg.indicators if cfg else [])),
            "blend_mode":    cfg.blend_mode if cfg else "",
            "phiai":         cfg.phiai_enabled if cfg else False,
            "initial_cap":   cfg.initial_capital if cfg else 0,
            "end_capital":   res.get("end_capital", ""),
            "total_return":  metrics.get("total_return", ""),
            "cagr":          metrics.get("cagr", ""),
            "sharpe":        metrics.get("sharpe", ""),
            "max_drawdown":  metrics.get("max_drawdown", ""),
            "profit_factor": metrics.get("profit_factor", ""),
            "win_rate":      metrics.get("win_rate", ""),
            "n_trades":      metrics.get("n_trades", 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    elif hasattr(obj, "__float__"):
        return float(obj)
    elif hasattr(obj, "__int__"):
        return int(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
