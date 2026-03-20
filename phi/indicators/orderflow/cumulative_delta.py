"""Cumulative volume delta indicator."""

from __future__ import annotations

import pandas as pd


def compute_cumulative_delta_signal(
    order_flow: pd.DataFrame,
    volume: pd.Series,
    window: int = 20,
    clip_value: float = 1.0,
) -> pd.Series:
    """Compute rolling cumulative volume delta normalized to [-1, 1]."""
    delta = order_flow["buy_volume"].astype(float) - order_flow["sell_volume"].astype(float)
    cum_delta = delta.rolling(window=int(window), min_periods=1).sum()
    total_volume = volume.astype(float).rolling(window=int(window), min_periods=1).sum().clip(lower=1e-10)
    norm = (cum_delta / total_volume).clip(-clip_value, clip_value)
    return (norm / max(clip_value, 1e-10)).fillna(0.0).rename("cumulative_delta_signal")
