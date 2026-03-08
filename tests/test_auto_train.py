from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from scripts import auto_train


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    base = {
        "tickers": ["SPY", "QQQ"],
        "timeframe": "1D",
        "years": 1,
        "end_date": "2024-12-31",
        "n_trials": 5,
        "windows": 2,
        "parallel": 1,
        "metric": "sharpe",
        "output_dir": str(tmp_path / "best_params"),
        "vendor": "yfinance",
        "force_refresh": False,
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_auto_train_runs_and_calls_dependencies(monkeypatch, tmp_path):
    args = _args(tmp_path, tickers=["spy"])
    monkeypatch.setattr(auto_train, "parse_args", lambda: args)

    sanitize_mock = MagicMock(return_value="SPY")
    fetch_mock = MagicMock(return_value=_sample_data())
    optimize_mock = MagicMock(return_value={"best_params": {"RSI": {"rsi_period": 14}}, "best_value": 1.23, "explanation": "ok"})
    save_mock = MagicMock()

    monkeypatch.setattr(auto_train, "sanitize_ticker", sanitize_mock)
    monkeypatch.setattr(auto_train, "fetch_and_cache", fetch_mock)
    monkeypatch.setattr(auto_train, "run_phiai_optimization", optimize_mock)
    monkeypatch.setattr(auto_train, "save_best_params", save_mock)

    auto_train.main()

    sanitize_mock.assert_called_once_with("spy")
    fetch_mock.assert_called_once()
    optimize_mock.assert_called_once()
    save_mock.assert_called_once()


def test_auto_train_handles_empty_data(monkeypatch, tmp_path):
    args = _args(tmp_path, tickers=["SPY"])
    monkeypatch.setattr(auto_train, "parse_args", lambda: args)

    monkeypatch.setattr(auto_train, "sanitize_ticker", lambda x: x)
    monkeypatch.setattr(auto_train, "fetch_and_cache", lambda **_: pd.DataFrame())
    optimize_mock = MagicMock()
    monkeypatch.setattr(auto_train, "run_phiai_optimization", optimize_mock)
    save_mock = MagicMock()
    monkeypatch.setattr(auto_train, "save_best_params", save_mock)

    auto_train.main()

    optimize_mock.assert_not_called()
    save_mock.assert_not_called()


def test_auto_train_skips_failed_ticker_and_continues(monkeypatch, tmp_path):
    args = _args(tmp_path, tickers=["BAD", "GOOD"])
    monkeypatch.setattr(auto_train, "parse_args", lambda: args)

    monkeypatch.setattr(auto_train, "sanitize_ticker", lambda x: x)

    def mock_fetch(**kwargs):
        if kwargs["symbol"] == "BAD":
            raise RuntimeError("boom")
        return _sample_data()

    monkeypatch.setattr(auto_train, "fetch_and_cache", mock_fetch)
    optimize_mock = MagicMock(return_value={"best_params": {}, "best_value": 0.0, "explanation": "ok"})
    save_mock = MagicMock()
    monkeypatch.setattr(auto_train, "run_phiai_optimization", optimize_mock)
    monkeypatch.setattr(auto_train, "save_best_params", save_mock)

    auto_train.main()

    optimize_mock.assert_called_once()
    save_mock.assert_called_once()


def test_auto_train_creates_output_directory(monkeypatch, tmp_path):
    args = _args(tmp_path, tickers=["SPY"], output_dir=str(tmp_path / "nested" / "best"))
    monkeypatch.setattr(auto_train, "parse_args", lambda: args)
    monkeypatch.setattr(auto_train, "sanitize_ticker", lambda x: x)
    monkeypatch.setattr(auto_train, "fetch_and_cache", lambda **_: _sample_data())
    monkeypatch.setattr(auto_train, "run_phiai_optimization", lambda **_: {"best_params": {}, "best_value": 0.0, "explanation": "ok"})
    monkeypatch.setattr(auto_train, "save_best_params", lambda *_, **__: None)

    auto_train.main()

    assert (tmp_path / "nested" / "best").exists()
