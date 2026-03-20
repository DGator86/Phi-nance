from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phi.regime.models.deep import DeepRegimeDetector


torch = pytest.importorskip("torch")


def _synthetic_ohlcv(rows: int = 140) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(123)
    rets = np.concatenate([rng.normal(0.001, 0.003, rows // 2), rng.normal(-0.001, 0.01, rows - rows // 2)])
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0.002, 0.001, rows)))
    low = close * (1 - np.abs(rng.normal(0.002, 0.001, rows)))
    open_ = close * (1 + rng.normal(0.0, 0.001, rows))
    volume = rng.integers(500_000, 2_000_000, rows)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _labels(data: pd.DataFrame) -> pd.Series:
    out = pd.Series("state_0", index=data.index)
    out.iloc[len(out) // 2 :] = "state_1"
    return out


def test_deep_detector_fit_predict_and_roundtrip(tmp_path) -> None:
    data = _synthetic_ohlcv()
    labels = _labels(data)

    detector = DeepRegimeDetector(
        model_type="lstm",
        seq_length=10,
        hidden_size=16,
        num_layers=1,
        epochs=2,
        batch_size=16,
        window=10,
    ).fit(data, labels=labels)

    pred = detector.predict(data)
    assert len(pred) == len(data)
    assert pred.dropna().astype(str).str.startswith("state_").all()

    path = tmp_path / "deep.pkl"
    detector.save(path)
    loaded = DeepRegimeDetector.load(path)
    pred_loaded = loaded.predict(data)
    assert len(pred_loaded) == len(pred)


def test_deep_detector_requires_labels() -> None:
    data = _synthetic_ohlcv(rows=60)
    with pytest.raises(ValueError, match="requires labels"):
        DeepRegimeDetector(model_type="lstm", epochs=1, seq_length=8, window=8).fit(data)
