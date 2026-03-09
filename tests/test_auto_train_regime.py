from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phi.backtest.direct import run_direct_backtest
from phi.regime import create_detector_from_params
from scripts import auto_train


def _sample_data(rows: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(range(100, 100 + rows), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


def test_create_detector_from_params() -> None:
    hmm = create_detector_from_params("hmm", {"n_regimes": 3, "covariance_type": "diag", "random_state": 7})
    km = create_detector_from_params("kmeans", {"n_regimes": 2, "random_state": 7})
    gmm = create_detector_from_params("gmm", {"n_regimes": 2, "random_state": 7})

    assert hmm.__class__.__name__ == "HMMRegimeDetector"
    assert km.__class__.__name__ == "ClusteringRegimeDetector"
    assert gmm.__class__.__name__ == "ClusteringRegimeDetector"


def test_auto_train_regime_optimize_saves_payload(monkeypatch, tmp_path: Path) -> None:
    args = argparse.Namespace(
        tickers=["SPY"],
        timeframe="1D",
        years=1,
        end_date="2024-12-31",
        n_trials=1,
        windows=2,
        parallel=1,
        metric="sharpe",
        output_dir=str(tmp_path),
        vendor="yfinance",
        force_refresh=False,
        verbose=False,
        regime_optimize=True,
        seed=42,
    )
    monkeypatch.setattr(auto_train, "parse_args", lambda: args)
    monkeypatch.setattr(auto_train, "sanitize_ticker", lambda x: x)
    monkeypatch.setattr(auto_train, "fetch_and_cache", lambda **_: _sample_data())

    auto_train.main()

    files = list(tmp_path.glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert "regime_detector" in payload
    assert "regime_boosts" in payload


def test_run_direct_backtest_accepts_detector_params() -> None:
    data = _sample_data()
    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"Buy & Hold": {"enabled": True, "params": {}}},
        blend_weights={"Buy & Hold": 1.0},
        blend_method="regime_weighted",
        regime_detector_params={
            "type": "kmeans",
            "params": {"n_regimes": 2, "feature_window": 10, "random_state": 42},
        },
        regime_boosts={"0": {"Buy & Hold": 1.0}, "1": {"Buy & Hold": 1.1}},
    )
    assert "portfolio_value" in results
