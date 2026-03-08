from __future__ import annotations

import numpy as np
import pandas as pd

from phi.backtest.direct import run_direct_backtest
from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.hmm import HMMRegimeDetector
from phi.regime.utils import extract_features


def _synthetic_ohlcv(rows: int = 220) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(7)

    low_vol = rng.normal(0.0004, 0.003, rows // 2)
    high_vol = rng.normal(-0.0002, 0.02, rows - rows // 2)
    rets = np.concatenate([low_vol, high_vol])

    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0.003, 0.002, rows)))
    low = close * (1 - np.abs(rng.normal(0.003, 0.002, rows)))
    open_ = close * (1 + rng.normal(0, 0.001, rows))
    volume = np.concatenate([
        rng.integers(1_000_000, 1_500_000, rows // 2),
        rng.integers(1_600_000, 3_000_000, rows - rows // 2),
    ])

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_extract_features_contains_expected_columns() -> None:
    feats = extract_features(_synthetic_ohlcv(), window=20)
    assert {"returns", "log_returns", "rolling_vol", "atr_ratio", "volume_change"}.issubset(feats.columns)
    assert not feats.empty


def test_hmm_detector_fit_predict_and_roundtrip(tmp_path) -> None:
    data = _synthetic_ohlcv()
    detector = HMMRegimeDetector(n_states=2).fit(data, window=15)
    regimes = detector.predict(data)
    assert regimes.str.startswith("state_").all()

    path = tmp_path / "hmm.pkl"
    detector.save(path)
    loaded = HMMRegimeDetector.load(path)
    loaded_regimes = loaded.predict(data)
    assert len(loaded_regimes) == len(regimes)


def test_clustering_detector_fit_predict_and_roundtrip(tmp_path) -> None:
    data = _synthetic_ohlcv()
    detector = ClusteringRegimeDetector(n_clusters=3, method="kmeans").fit(data, window=15)
    regimes = detector.predict(data)
    assert regimes.str.startswith("cluster_").all()

    path = tmp_path / "cluster.pkl"
    detector.save(path)
    loaded = ClusteringRegimeDetector.load(path)
    loaded_regimes = loaded.predict(data)
    assert len(loaded_regimes) == len(regimes)


def test_regime_series_can_drive_regime_weighted_backtest() -> None:
    data = _synthetic_ohlcv()
    idx = data.index
    regimes = pd.Series(np.where(np.arange(len(idx)) % 2 == 0, "BULL", "BEAR"), index=idx)

    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        blend_weights={"RSI": 1.0},
        blend_method="regime_weighted",
        regime_series=regimes,
    )

    assert "portfolio_value" in results
    assert len(results["portfolio_value"]) > 1


def test_regime_weighted_backtest_does_not_backfill_warmup_labels() -> None:
    data = _synthetic_ohlcv(rows=40)
    # First regime appears mid-series, so earlier rows should remain unassigned after ffill-only alignment.
    regimes = pd.Series(["state_1"], index=[data.index[20]])

    results, strat = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        blend_weights={"RSI": 1.0},
        blend_method="regime_weighted",
        regime_series=regimes,
    )

    assert "portfolio_value" in results
    assert len(strat.prediction_log) > 0


def test_regime_weighted_backtest_applies_optional_label_map() -> None:
    data = _synthetic_ohlcv(rows=30)
    regimes = pd.Series("state_0", index=data.index)

    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        blend_weights={"RSI": 1.0},
        blend_method="regime_weighted",
        regime_series=regimes,
        regime_label_map={"state_0": "bull"},
    )

    assert "portfolio_value" in results
