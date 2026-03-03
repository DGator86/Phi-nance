"""
phinance.storage.local
=======================

LocalStorage — low-level file I/O for backtest runs.

On-disk layout (per run)::

    runs/
      {run_id}/
        config.json
        results.json
        trades.csv       (optional)

This module deals only with raw JSON / CSV I/O and path management.
Higher-level logic (run IDs, history queries) lives in ``run_history.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.utils.logging import get_logger

logger = get_logger(__name__)


class LocalStorage:
    """Low-level file I/O for backtest run artefacts.

    Parameters
    ----------
    root : Path, optional
        Root directory for all run data.
        Defaults to ``<project_root>/runs``.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            _project_root = Path(__file__).resolve().parent.parent.parent
            root = _project_root / "runs"
        self.root: Path = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Run directory helpers ─────────────────────────────────────────────────

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def ensure_run_dir(self, run_id: str) -> Path:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Config I/O ────────────────────────────────────────────────────────────

    def write_config(self, run_id: str, config: Dict[str, Any]) -> None:
        p = self.ensure_run_dir(run_id) / "config.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        logger.debug("Wrote config for run %s", run_id)

    def read_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        p = self.run_dir(run_id) / "config.json"
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read config for %s: %s", run_id, exc)
            return None

    # ── Results I/O ───────────────────────────────────────────────────────────

    def write_results(self, run_id: str, results: Dict[str, Any]) -> None:
        p = self.ensure_run_dir(run_id) / "results.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.debug("Wrote results for run %s", run_id)

    def read_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        p = self.run_dir(run_id) / "results.json"
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read results for %s: %s", run_id, exc)
            return None

    # ── Trades I/O ────────────────────────────────────────────────────────────

    def write_trades(self, run_id: str, trades: pd.DataFrame) -> None:
        if trades is None or trades.empty:
            return
        p = self.ensure_run_dir(run_id) / "trades.csv"
        trades.to_csv(p, index=False)
        logger.debug("Wrote %d trades for run %s", len(trades), run_id)

    def read_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        p = self.run_dir(run_id) / "trades.csv"
        if not p.exists():
            return None
        try:
            return pd.read_csv(p)
        except Exception as exc:
            logger.warning("Failed to read trades for %s: %s", run_id, exc)
            return None

    # ── Directory listing ─────────────────────────────────────────────────────

    def list_run_ids(self) -> List[str]:
        """Return all run_ids sorted newest first."""
        if not self.root.exists():
            return []
        return sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()],
            reverse=True,
        )
