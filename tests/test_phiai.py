from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from phi.run_config import RunConfig


def _sample_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0.1, 1.0, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1000),
        },
        index=idx,
    )


def test_run_phiai_optimization_returns_best_params(tmp_path: Path, monkeypatch):
    from phi.phiai import auto_tune as at

    monkeypatch.setattr(at, "_best_params_dir", lambda: tmp_path)

    ohlcv = _sample_ohlcv()
    indicators = {
        "RSI": {"enabled": True, "auto_tune": True, "params": {"rsi_period": 14}},
        "MACD": {"enabled": True, "auto_tune": False, "params": {"fast_period": 12}},
    }

    result = at.run_phiai_optimization(ohlcv=ohlcv, indicators_config=indicators, n_trials=6, walk_forward_windows=2)

    assert isinstance(result["best_params"], dict)
    assert "optimized_indicators" in result
    assert not math.isnan(result["best_value"]), "best_value should not be NaN"
    assert (tmp_path / f"{result['dataset_id']}.json").exists()


def test_save_and_load_best_params(tmp_path: Path, monkeypatch):
    from phi.phiai import auto_tune as at

    monkeypatch.setattr(at, "_best_params_dir", lambda: tmp_path)

    expected = {"RSI": {"rsi_period": 7, "oversold": 30, "overbought": 70}}
    at.save_best_params(expected, dataset_id="unit_ds", best_value=0.12, metric="sharpe")

    payload = at.load_best_params("unit_ds")
    assert payload is not None
    assert payload["best_params"] == expected
    assert payload["metric"] == "sharpe"


def test_run_config_overrides_are_applied(tmp_path: Path, monkeypatch):
    from phi.phiai import auto_tune as at

    monkeypatch.setattr(at, "_best_params_dir", lambda: tmp_path)

    captured = {}

    class DummyStudy:
        best_value = 0.33
        best_trial = type("Trial", (), {"user_attrs": {"best_params": {"RSI": {"rsi_period": 7}}}})()

        def optimize(self, objective, n_trials, n_jobs):
            captured["n_trials"] = n_trials
            captured["n_jobs"] = n_jobs
            trial = type("T", (), {"suggest_categorical": lambda self, name, values: values[0], "set_user_attr": lambda *a, **k: None})()
            objective(trial)

    monkeypatch.setattr(at.optuna, "create_study", lambda direction, sampler: DummyStudy())

    cfg = RunConfig(
        dataset_id="cfg_ds",
        symbols=["SPY"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        indicators={"RSI": {"enabled": True, "params": {}}},
        blend_method="weighted_sum",
        blend_weights={"RSI": 1.0},
        phiai_enabled=True,
        phiai_n_trials=4,
        phiai_walk_forward_windows=2,
        phiai_parallel_jobs=2,
    )

    ohlcv = _sample_ohlcv(90)
    indicators = {"RSI": {"enabled": True, "auto_tune": True, "params": {"rsi_period": 14}}}

    result = at.run_phiai_optimization(ohlcv=ohlcv, indicators_config=indicators, run_config=cfg)

    assert captured["n_trials"] == 4
    assert captured["n_jobs"] == 2
    assert result["dataset_id"] == "cfg_ds"


def test_infer_periods_per_year_intraday_index() -> None:
    from phi.phiai import auto_tune as at

    idx = pd.date_range("2024-01-02 09:30", periods=60, freq="1min")
    periods = at._infer_periods_per_year(idx)

    assert periods > 252.0
