"""
phinance.strategies.rsi
=======================

RSI (Relative Strength Index) — Wilder's original smoothed-average method.

References
----------
* Wilder, J.W. (1978) *New Concepts in Technical Trading Systems*
* stockstats (jealous/stockstats) — RSI implementation
* Stock.Indicators (.NET) — Rsi() series method

Formula
-------
  delta  = close.diff()
  gain   = delta.clip(lower=0)
  loss   = (-delta).clip(lower=0)

  Wilder smoothing (SMMA):
    avg_gain[t] = (avg_gain[t-1] * (n-1) + gain[t]) / n
    avg_loss[t] = (avg_loss[t-1] * (n-1) + loss[t]) / n

  RS    = avg_gain / avg_loss
  RSI   = 100 – 100 / (1 + RS)

Signal convention
-----------------
  +1 → deeply oversold  (RSI ≤ oversold  threshold — expect mean reversion up)
  −1 → deeply overbought(RSI ≥ overbought threshold — expect mean reversion dn)
   0 → neutral mid-zone
  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class RSIIndicator(BaseIndicator):
    """RSI mean-reversion signal using Wilder's SMMA smoothing.

    Parameters
    ----------
    period     : int   — lookback window (default 14)
    oversold   : float — RSI ≤ this  → +1  (default 30)
    overbought : float — RSI ≥ this  → −1  (default 70)
    """

    name = "RSI"
    default_params = {"period": 14, "oversold": 30, "overbought": 70}
    param_grid = {
        "period":     [7, 9, 14, 21],
        "oversold":   [25, 30, 35],
        "overbought": [65, 70, 75],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder smoothing: seed with SMA of first `period` bars, then SMMA
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # When avg_loss == 0 (pure uptrend), RSI = 100; cap RS at a large number
        rs  = avg_gain / avg_loss.where(avg_loss != 0, other=1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Map RSI → signal in [−1, 1]
        # Linearly interpolate: oversold→+1, midpoint(50)→0, overbought→−1
        mid    = 50.0
        signal = pd.Series(0.0, index=rsi.index)
        # Oversold zone: RSI in [0, oversold] → signal in [+1, 0]
        in_os = rsi <= oversold
        signal[in_os] = ((oversold - rsi[in_os]) / oversold).clip(0, 1)
        # Overbought zone: RSI in [overbought, 100] → signal in [0, −1]
        in_ob = rsi >= overbought
        signal[in_ob] = -((rsi[in_ob] - overbought) / (100 - overbought)).clip(0, 1)
        # Mid zone: linear interpolation
        mid_zone = ~in_os & ~in_ob
        signal[mid_zone] = -(rsi[mid_zone] - mid) / (overbought - mid)

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("RSI")
