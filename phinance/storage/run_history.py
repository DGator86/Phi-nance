"""
phinance.storage.run_history
=============================

RunHistory — high-level API for managing backtest run storage.

Usage
-----
    from phinance.storage import RunHistory
    from phinance.config.run_config import RunConfig

    history = RunHistory()

    # Persist a new run
    run_id = history.create_run(config)
    history.save_results(run_id, results_dict, trades_df)

    # Query history
    runs = history.list_runs()
    run = history.load_run(run_id)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.config.run_config import RunConfig
from phinance.storage.local import LocalStorage
from phinance.storage.models import StoredRun
from phinance.exceptions import RunNotFoundError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


def _new_run_id() -> str:
    """Generate a unique run ID: ``YYYYMMDD_HHMMSS_{hex8}``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class RunHistory:
    """High-level backtest run storage manager.

    Parameters
    ----------
    root : Path, optional
        Root directory for run storage.
        Defaults to ``<project_root>/runs``.

    Example
    -------
        history = RunHistory()
        run_id = history.create_run(cfg)
        history.save_results(run_id, {"total_return": 0.12, ...})
        runs = history.list_runs()        # newest first
        run  = history.load_run(run_id)   # StoredRun
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._storage = LocalStorage(root=root)

    # ── Create ────────────────────────────────────────────────────────────────

    def create_run(self, config: RunConfig) -> str:
        """Persist a new run config and return its run_id.

        Parameters
        ----------
        config : RunConfig

        Returns
        -------
        str — unique run_id
        """
        run_id = _new_run_id()
        self._storage.write_config(run_id, config.to_dict())
        logger.info("Created run %s: %s", run_id, config)
        return run_id

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_results(
        self,
        run_id: str,
        results: Dict[str, Any],
        trades: Optional[pd.DataFrame] = None,
    ) -> None:
        """Persist results (and optionally trades) for an existing run.

        Parameters
        ----------
        run_id  : str
        results : dict — backtest metrics
        trades  : pd.DataFrame, optional
        """
        self._storage.write_results(run_id, results)
        if trades is not None and not trades.empty:
            self._storage.write_trades(run_id, trades)
        logger.info("Saved results for run %s", run_id)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_run(self, run_id: str) -> StoredRun:
        """Load a complete run record from disk.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        StoredRun

        Raises
        ------
        RunNotFoundError
        """
        config = self._storage.read_config(run_id)
        if config is None:
            raise RunNotFoundError(f"Run '{run_id}' not found.")
        results = self._storage.read_results(run_id) or {}
        trades = self._storage.read_trades(run_id)
        return StoredRun(run_id=run_id, config=config, results=results, trades=trades)

    def load_config(self, run_id: str) -> Optional[RunConfig]:
        """Load only the config for a run."""
        d = self._storage.read_config(run_id)
        if d is None:
            return None
        return RunConfig.from_dict(d)

    def load_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load only the results dict for a run."""
        return self._storage.read_results(run_id)

    def load_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        """Load only the trades DataFrame for a run."""
        return self._storage.read_trades(run_id)

    # ── List ──────────────────────────────────────────────────────────────────

    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return basic metadata for the most recent *limit* runs.

        Parameters
        ----------
        limit : int — maximum number of runs to return (default 100)

        Returns
        -------
        list of dicts — each has ``run_id``, ``config``, ``results`` keys
        """
        run_ids = self._storage.list_run_ids()[:limit]
        records: List[Dict[str, Any]] = []
        for rid in run_ids:
            cfg = self._storage.read_config(rid)
            res = self._storage.read_results(rid)
            records.append({
                "run_id":  rid,
                "config":  cfg or {},
                "results": res or {},
            })
        return records

    def list_stored_runs(self, limit: int = 100) -> List[StoredRun]:
        """Return fully-materialised ``StoredRun`` objects (newest first).

        This is more expensive than ``list_runs()`` because it also loads
        trades from disk.
        """
        run_ids = self._storage.list_run_ids()[:limit]
        runs: List[StoredRun] = []
        for rid in run_ids:
            try:
                runs.append(self.load_run(rid))
            except Exception as exc:
                logger.warning("Skipping run %s: %s", rid, exc)
        return runs
