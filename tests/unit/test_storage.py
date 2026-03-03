"""
tests/unit/test_storage.py
===========================

Unit tests for phinance.storage:
  - LocalStorage I/O (config, results, trades)
  - RunHistory CRUD operations
  - StoredRun properties and serialisation
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import pandas as pd
import pytest

from phinance.storage.local import LocalStorage
from phinance.storage.models import StoredRun
from phinance.storage.run_history import RunHistory, _new_run_id
from phinance.config.run_config import RunConfig
from phinance.exceptions import RunNotFoundError


# ─────────────────────────────────────────────────────────────────────────────
#  _new_run_id
# ─────────────────────────────────────────────────────────────────────────────

class TestNewRunId:

    def test_format(self):
        rid = _new_run_id()
        # format: YYYYMMDD_HHMMSS_xxxxxxxx
        parts = rid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8   # date
        assert len(parts[1]) == 6   # time
        assert len(parts[2]) == 8   # hex

    def test_uniqueness(self):
        ids = {_new_run_id() for _ in range(20)}
        assert len(ids) == 20  # no duplicates


# ─────────────────────────────────────────────────────────────────────────────
#  LocalStorage
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalStorage:

    def test_creates_root_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs_test"
            ls = LocalStorage(root=root)
            assert root.exists()

    def test_write_and_read_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            ls.write_config("run1", {"symbols": ["SPY"], "timeframe": "1D"})
            result = ls.read_config("run1")
            assert result["symbols"] == ["SPY"]
            assert result["timeframe"] == "1D"

    def test_read_missing_config_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            assert ls.read_config("nonexistent") is None

    def test_write_and_read_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            results = {"sharpe": 1.5, "total_return": 0.12, "cagr": 0.10}
            ls.write_results("run1", results)
            loaded = ls.read_results("run1")
            assert abs(loaded["sharpe"] - 1.5) < 1e-9

    def test_read_missing_results_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            assert ls.read_results("nonexistent") is None

    def test_write_and_read_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            trades = pd.DataFrame({
                "entry_date": ["2023-01-05"],
                "exit_date":  ["2023-01-10"],
                "pnl":        [250.0],
            })
            ls.write_trades("run1", trades)
            loaded = ls.read_trades("run1")
            assert loaded is not None
            assert len(loaded) == 1
            assert loaded["pnl"].iloc[0] == 250.0

    def test_read_missing_trades_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            assert ls.read_trades("nonexistent") is None

    def test_write_empty_trades_skips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            ls.write_trades("run1", pd.DataFrame())
            # No file created for empty trades
            assert not (Path(tmpdir) / "run1" / "trades.csv").exists()

    def test_list_run_ids_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            assert ls.list_run_ids() == []

    def test_list_run_ids_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            for rid in ["20230101_001", "20240101_001", "20220101_001"]:
                ls.write_config(rid, {"x": 1})
            ids = ls.list_run_ids()
            assert ids[0] > ids[1] > ids[2]

    def test_json_serialisable_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ls = LocalStorage(root=Path(tmpdir))
            config = {"symbols": ["SPY"], "initial_capital": 100_000.0}
            ls.write_config("r1", config)
            path = ls.run_dir("r1") / "config.json"
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["symbols"] == ["SPY"]


# ─────────────────────────────────────────────────────────────────────────────
#  StoredRun
# ─────────────────────────────────────────────────────────────────────────────

class TestStoredRun:

    def test_properties_with_data(self):
        run = StoredRun(
            run_id  = "20240101_test",
            config  = {"symbols": ["AAPL", "MSFT"], "timeframe": "1D"},
            results = {"total_return": 0.15, "sharpe": 1.8, "cagr": 0.14,
                       "max_drawdown": 0.08},
        )
        assert run.symbols == ["AAPL", "MSFT"]
        assert abs(run.total_return - 0.15) < 1e-9
        assert abs(run.sharpe       - 1.8)  < 1e-9
        assert abs(run.cagr         - 0.14) < 1e-9
        assert abs(run.max_drawdown - 0.08) < 1e-9

    def test_properties_empty_config(self):
        run = StoredRun(run_id="r1")
        assert run.symbols == []
        assert run.total_return == 0.0
        assert run.sharpe       == 0.0

    def test_summary_string(self):
        run = StoredRun(
            run_id  = "20240101_120000_abc",
            config  = {"symbols": ["SPY"], "timeframe": "1D",
                       "start_date": "2022-01-01", "end_date": "2024-12-31"},
            results = {"total_return": 0.18, "sharpe": 1.4, "max_drawdown": 0.09},
        )
        s = run.summary()
        assert "SPY" in s
        assert "1D"  in s

    def test_repr(self):
        run = StoredRun(run_id="test_run", config={"symbols": ["GOOG"]})
        r = repr(run)
        assert "test_run" in r


# ─────────────────────────────────────────────────────────────────────────────
#  RunHistory
# ─────────────────────────────────────────────────────────────────────────────

class TestRunHistory:

    def _make_cfg(self, **kw) -> RunConfig:
        defaults = dict(
            symbols         = ["SPY"],
            start_date      = "2023-01-01",
            end_date        = "2023-12-31",
            timeframe       = "1D",
            vendor          = "yfinance",
            initial_capital = 100_000.0,
        )
        defaults.update(kw)
        return RunConfig(**defaults)

    def test_create_run_returns_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            run_id  = history.create_run(self._make_cfg())
            assert isinstance(run_id, str)
            assert len(run_id) > 0

    def test_save_and_load_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            run_id  = history.create_run(self._make_cfg())
            history.save_results(run_id, {"sharpe": 2.0, "total_return": 0.22})

            results = history.load_results(run_id)
            assert results is not None
            assert abs(results["sharpe"] - 2.0) < 1e-9

    def test_load_run_returns_stored_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            cfg    = self._make_cfg(symbols=["QQQ"])
            run_id = history.create_run(cfg)
            history.save_results(run_id, {"cagr": 0.18})

            run = history.load_run(run_id)
            assert isinstance(run, StoredRun)
            assert run.config.get("symbols") == ["QQQ"]
            assert run.results["cagr"] == 0.18

    def test_load_missing_run_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            with pytest.raises(RunNotFoundError):
                history.load_run("nonexistent_run")

    def test_list_runs_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            ids = []
            for sym in ["SPY", "QQQ", "AAPL"]:
                ids.append(history.create_run(self._make_cfg(symbols=[sym])))

            runs = history.list_runs()
            assert len(runs) >= 3
            run_ids = [r["run_id"] for r in runs]
            # Sorted newest first
            assert run_ids == sorted(run_ids, reverse=True)

    def test_list_runs_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            for _ in range(5):
                history.create_run(self._make_cfg())
            runs = history.list_runs(limit=3)
            assert len(runs) <= 3

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            cfg    = self._make_cfg(blend_method="voting")
            run_id = history.create_run(cfg)

            loaded_cfg = history.load_config(run_id)
            assert loaded_cfg is not None
            assert loaded_cfg.blend_method == "voting"

    def test_save_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            run_id  = history.create_run(self._make_cfg())
            trades  = pd.DataFrame({
                "entry_date": ["2023-01-05", "2023-02-10"],
                "exit_date":  ["2023-01-12", "2023-02-18"],
                "pnl":        [300.0, -120.0],
            })
            history.save_results(run_id, {}, trades=trades)

            loaded = history.load_trades(run_id)
            assert loaded is not None
            assert len(loaded) == 2

    def test_list_stored_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = RunHistory(root=Path(tmpdir))
            for _ in range(3):
                run_id = history.create_run(self._make_cfg())
                history.save_results(run_id, {"sharpe": 1.0})

            stored_runs = history.list_stored_runs()
            assert len(stored_runs) == 3
            assert all(isinstance(r, StoredRun) for r in stored_runs)
