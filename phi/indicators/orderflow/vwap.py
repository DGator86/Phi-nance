"""VWAP order flow indicator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap_series(ohlcv: pd.DataFrame) -> pd.Series:
    """Compute cumulative VWAP across the provided series."""
    tp = (ohlcv["high"].astype(float) + ohlcv["low"].astype(float) + ohlcv["close"].astype(float)) / 3.0
    vol = ohlcv["volume"].astype(float).clip(lower=0.0)
    tpv = (tp * vol).cumsum()
    cum_vol = vol.cumsum().clip(lower=1e-10)
    return (tpv / cum_vol).rename("vwap")


def compute_vwap_signal(ohlcv: pd.DataFrame, atr_period: int = 14, clip_value: float = 1.0) -> pd.Series:
    """Compute normalized VWAP deviation signal in [-1, 1]."""
    close = ohlcv["close"].astype(float)
    vwap = compute_vwap_series(ohlcv)

    tr = pd.concat(
        [
            ohlcv["high"].astype(float) - ohlcv["low"].astype(float),
            (ohlcv["high"].astype(float) - close.shift(1)).abs(),
            (ohlcv["low"].astype(float) - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=int(atr_period), adjust=False).mean().clip(lower=1e-10)
    signal = ((close - vwap) / atr).clip(-clip_value, clip_value)
    return (signal / max(clip_value, 1e-10)).fillna(0.0).rename("vwap_signal")
