from __future__ import annotations

import numpy as np
import pandas as pd

from phinance.ml.data import FeatureScaler, MarketSequenceDataset, prepare_market_features


def _frame(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100.0 + idx * 0.1 + np.sin(idx / 5.0)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000 + idx * 100,
        }
    )


def test_prepare_market_features_adds_engineered_columns() -> None:
    features = prepare_market_features(_frame())
    for col in ["return_1d", "realized_vol_20", "rsi_14", "macd", "bollinger_pct_b", "volume_z_20"]:
        assert col in features.columns


def test_scaler_roundtrip() -> None:
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
    scaler = FeatureScaler.fit(values)
    payload = scaler.to_dict()
    restored = FeatureScaler.from_dict(payload)
    transformed = restored.transform(values)
    assert transformed.shape == values.shape


def test_sequence_dataset_shapes() -> None:
    ds = MarketSequenceDataset([_frame(140)], sequence_length=20, fit_scaler=True)
    x, y = ds[0]
    assert x.shape[0] == 20
    assert x.shape[1] == len(ds.feature_columns)
    assert y.ndim == 0
