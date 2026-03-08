from __future__ import annotations

from datetime import date

import pandas as pd

from phi.phiai import auto_tune
from phi.run_config import RunConfig


def _ohlcv(rows: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(range(100, 100 + rows), index=idx, dtype=float)
    return pd.DataFrame({"close": close, "open": close, "high": close + 1, "low": close - 1, "volume": 1_000})


def test_save_and_load_best_params_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_tune, "_best_params_dir", lambda: tmp_path)

    payload = {"RSI": {"period": 14}}
    saved = auto_tune.save_best_params(payload, dataset_id="demo", best_value=1.25, metric="sharpe")

    assert saved.exists()
    loaded = auto_tune.load_best_params("demo")
    assert loaded["best_params"] == payload
    assert loaded["best_value"] == 1.25


def test_run_phiai_returns_no_changes_when_no_auto_tune(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_tune, "_best_params_dir", lambda: tmp_path)

    indicators = {"RSI": {"enabled": True, "auto_tune": False, "params": {"period": 14}}}
    cfg = RunConfig(symbols=["SPY"], start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))

    result = auto_tune.run_phiai_optimization(_ohlcv(), indicators, run_config=cfg, n_trials=1)

    assert result["study"] is None
    assert result["best_params"] == {}
    assert "made no changes" in result["explanation"]
    assert result["optimized_indicators"] == indicators
