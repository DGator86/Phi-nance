"""
phinance.strategies.atr
========================

ATR Trend-Strength indicator (Normalised ATR as a regime signal).

References
----------
* Wilder, J.W. (1978) ATR definition
* stockstats: atr / tr  (SMMA smoothing variant)
* arkochhar/Technical-Indicators: ATR()
* Stock.Indicators (.NET): Atr()

Formula
-------
  True Range (TR) = max of:
    |high − low|
    |high − prev_close|
    |low  − prev_close|

  ATR = SMMA(TR, n)            (Wilder's smoothed moving average)
      ≡ EMA(TR, span=n, alpha=1/n)

  Normalised ATR = ATR / close  (as % of price, scale-invariant)

  The *relative* ATR is then z-scored over a rolling look-back window
  to produce a signal:
    high rel-ATR → HIGHVOL environment
    low  rel-ATR → LOWVOL  environment

  Signal convention (trend-strength):
    +1 → volatility expanding above historical mean (favour breakout/momentum)
    −1 → volatility compressing below historical mean (favour mean-reversion)
     0 → typical volatility level
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class ATRIndicator(BaseIndicator):
    """Normalised ATR volatility-regime signal.

    Parameters
    ----------
    period      : int   — ATR smoothing window (default 14)
    lookback    : int   — historical window for z-scoring (default 50)
    z_threshold : float — clip z at ±z_threshold (default 2.0)
    """

    name = "ATR"
    default_params = {"period": 14, "lookback": 50, "z_threshold": 2.0}
    param_grid = {
        "period":      [7, 10, 14, 20],
        "lookback":    [30, 50, 100],
        "z_threshold": [1.5, 2.0, 2.5],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int       = 14,
        lookback: int     = 50,
        z_threshold: float = 2.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev  = close.shift(1)

        tr  = pd.concat(
            [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
            axis=1,
        ).max(axis=1)

        # Wilder SMMA ≡ EMA with alpha = 1/period
        atr     = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rel_atr = atr / close.replace(0.0, np.nan)  # scale-invariant

        # Z-score over rolling lookback
        mu  = rel_atr.rolling(window=lookback, min_periods=lookback).mean()
        std = rel_atr.rolling(window=lookback, min_periods=lookback).std(ddof=1).replace(0.0, np.nan)
        z   = ((rel_atr - mu) / std).clip(-z_threshold, z_threshold)

        signal = (z / z_threshold)   # +1 = high vol, −1 = low vol

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("ATR")
