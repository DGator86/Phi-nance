"""Data preparation utilities for deep regime detectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phi.regime.utils import extract_features


FeatureExtractor = Callable[[pd.DataFrame], pd.DataFrame]


def create_sequences(features: pd.DataFrame, labels: pd.Series, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length sequence tensors for supervised training.

    Args:
        features: Feature matrix indexed by timestamp.
        labels: Label series aligned to ``features`` index.
        seq_length: Number of bars in each input sequence.

    Returns:
        Tuple of ``(X, y)`` where ``X`` has shape ``(n_samples, seq_length, n_features)``.
    """
    if seq_length < 2:
        raise ValueError("seq_length must be >= 2")
    if features.empty:
        raise ValueError("features must not be empty")

    aligned_labels = labels.reindex(features.index)
    valid = aligned_labels.notna()
    feats = features.loc[valid]
    lbls = aligned_labels.loc[valid]

    if len(feats) < seq_length:
        raise ValueError("Not enough rows to build sequences")

    x_rows: list[np.ndarray] = []
    y_rows: list[Any] = []
    feat_values = feats.to_numpy(dtype=np.float32)
    label_values = lbls.to_numpy()

    for end in range(seq_length - 1, len(feat_values)):
        start = end - seq_length + 1
        x_rows.append(feat_values[start : end + 1])
        y_rows.append(label_values[end])

    return np.stack(x_rows), np.asarray(y_rows)


def prepare_data_for_training(
    ohlcv: pd.DataFrame,
    detector_params: dict[str, Any],
    feature_extractor: FeatureExtractor | None = None,
    labels: pd.Series | None = None,
) -> dict[str, Any]:
    """Prepare feature tensors, scaler, and train/validation splits for deep training."""
    extractor = feature_extractor or (lambda df: extract_features(df, window=int(detector_params.get("window", 20))))
    features = extractor(ohlcv)

    if labels is None:
        raise ValueError("labels are required for supervised deep regime training")

    labels = labels.reindex(features.index)
    seq_length = int(detector_params.get("seq_length", 20))
    val_split = float(detector_params.get("val_split", 0.2))

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    split_idx = max(seq_length + 1, int(len(features) * (1.0 - val_split)))
    train_features = features.iloc[:split_idx]

    scaler = StandardScaler()
    scaler.fit(train_features.to_numpy(dtype=np.float32))
    scaled = pd.DataFrame(
        scaler.transform(features.to_numpy(dtype=np.float32)),
        index=features.index,
        columns=features.columns,
    )

    x_all, y_all_raw = create_sequences(scaled, labels, seq_length=seq_length)
    y_encoded, unique_labels = pd.factorize(pd.Series(y_all_raw), sort=True)

    y_index = scaled.index[seq_length - 1 :]
    train_mask = y_index < scaled.index[split_idx]

    return {
        "features": features,
        "scaled_features": scaled,
        "X_train": x_all[train_mask],
        "y_train": y_encoded[train_mask],
        "X_val": x_all[~train_mask],
        "y_val": y_encoded[~train_mask],
        "labels": list(map(str, unique_labels)),
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "sequence_index": y_index,
    }
