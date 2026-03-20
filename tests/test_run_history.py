from __future__ import annotations

from datetime import date

import pandas as pd

from phi.run_config import RunConfig, RunHistory


def _config() -> RunConfig:
    return RunConfig(
        symbols=["SPY"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 10),
        indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
        blend_weights={"RSI": 1.0},
    )


def test_run_history_roundtrip(tmp_path):
    history = RunHistory(root=tmp_path)
    run_id = history.create_run(_config())

    history.save_results(run_id, {"total_return": 0.1}, trades=pd.DataFrame({"pnl": [1, 2]}))

    cfg = history.load_config(run_id)
    res = history.load_results(run_id)
    trades = history.load_trades(run_id)

    assert cfg is not None
    assert res["total_return"] == 0.1
    assert list(trades.columns) == ["pnl"]


def test_run_history_list_runs_includes_saved_run(tmp_path):
    history = RunHistory(root=tmp_path)
    run_id = history.create_run(_config())
    history.save_results(run_id, {"total_return": 0.2})

    listed = history.list_runs()

    assert any(row["run_id"] == run_id for row in listed)
