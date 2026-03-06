"""Dataset and feature engineering for market transformer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class FeatureScaler:
    """Simple feature-wise standard scaler for NumPy arrays."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "FeatureScaler":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))


def _validate_price_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_market_features(
    df: pd.DataFrame,
    *,
    include_regime: bool = False,
    regime_column: str = "regime",
) -> pd.DataFrame:
    """Compute normalized-ready engineered market features."""
    _validate_price_frame(df)

    features = df.copy()
    close = features["close"].astype(float)

    features["return_1d"] = close.pct_change().fillna(0.0)
    features["log_return"] = np.log(close).diff().fillna(0.0)
    features["realized_vol_20"] = features["return_1d"].rolling(20, min_periods=2).std().fillna(0.0)

    delta = close.diff().fillna(0.0)
    gains = delta.clip(lower=0.0).rolling(14, min_periods=2).mean()
    losses = (-delta.clip(upper=0.0)).rolling(14, min_periods=2).mean()
    rs = gains / losses.replace(0.0, np.nan)
    features["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd
    features["macd_signal"] = signal

    rolling_mean = close.rolling(20, min_periods=2).mean()
    rolling_std = close.rolling(20, min_periods=2).std().replace(0.0, np.nan)
    features["bollinger_pct_b"] = ((close - (rolling_mean - 2.0 * rolling_std)) / (4.0 * rolling_std)).fillna(0.5)

    features["volume_z_20"] = (
        (features["volume"] - features["volume"].rolling(20, min_periods=2).mean())
        / features["volume"].rolling(20, min_periods=2).std().replace(0.0, np.nan)
    ).fillna(0.0)

    if include_regime and regime_column in features:
        dummies = pd.get_dummies(features[regime_column], prefix="regime")
        features = pd.concat([features, dummies], axis=1)

    return features


class MarketSequenceDataset(Dataset):
    """Sliding-window dataset for transformer next-day return regression."""

    def __init__(
        self,
        frames: Iterable[pd.DataFrame],
        *,
        sequence_length: int = 60,
        feature_columns: list[str] | None = None,
        scaler: FeatureScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")

        sequences: list[np.ndarray] = []
        targets: list[float] = []

        for frame in frames:
            engineered = prepare_market_features(frame)
            cols = feature_columns or [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "return_1d",
                "realized_vol_20",
                "rsi_14",
                "macd",
                "macd_signal",
                "bollinger_pct_b",
                "volume_z_20",
            ]
            matrix = engineered[cols].astype(float).to_numpy()
            returns = engineered["return_1d"].to_numpy()
            if len(matrix) <= sequence_length:
                continue
            for idx in range(sequence_length, len(matrix) - 1):
                sequences.append(matrix[idx - sequence_length : idx])
                targets.append(float(returns[idx + 1]))

        if not sequences:
            raise ValueError("No sequences created. Provide longer historical series.")

        tensor = np.asarray(sequences, dtype=np.float32)
        self.feature_columns = cols
        self.sequence_length = sequence_length

        if scaler is not None:
            self.scaler = scaler
        elif fit_scaler:
            self.scaler = FeatureScaler.fit(tensor.reshape(-1, tensor.shape[-1]))
        else:
            self.scaler = FeatureScaler(mean=np.zeros(tensor.shape[-1]), std=np.ones(tensor.shape[-1]))

        tensor = self.scaler.transform(tensor)
        self._x = torch.tensor(tensor, dtype=torch.float32)
        self._y = torch.tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]
