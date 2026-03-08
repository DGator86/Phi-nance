"""Liquidity-oriented indicator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidity_signal(
    ohlcv: pd.DataFrame,
    order_flow: pd.DataFrame,
    amihud_scale: float = 1e6,
    window: int = 20,
) -> pd.Series:
    """Compute liquidity signal using spread (if available) and Amihud proxy."""
    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=1.0)
    ret = close.pct_change().abs().fillna(0.0)

    spread = order_flow["spread"].astype(float)
    spread_proxy = ((ohlcv["high"].astype(float) - ohlcv["low"].astype(float)) / close.replace(0.0, np.nan)).fillna(0.0)
    eff_spread = spread.where(spread > 0.0, spread_proxy)

    amihud = (ret / volume) * float(amihud_scale)
    amihud_z = (amihud - amihud.rolling(window, min_periods=max(2, window // 3)).mean())
    amihud_std = amihud.rolling(window, min_periods=max(2, window // 3)).std().clip(lower=1e-10)
    amihud_z = (amihud_z / amihud_std).fillna(0.0)

    liq_raw = -eff_spread * 200.0 - amihud_z + np.log1p(volume / volume.rolling(window, min_periods=1).mean().clip(lower=1.0))
    return np.tanh(liq_raw / 2.0).rename("liquidity_signal")
