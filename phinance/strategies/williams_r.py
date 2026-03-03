"""
phinance.strategies.williams_r
================================

Williams %R — momentum oscillator by Larry Williams.

References
----------
* Williams, L. (1973) original %R formula
* stockstats: wr  (WR implementation)
* arkochhar/Technical-Indicators: WilliamsR()
* Stock.Indicators (.NET): WilliamsR()

Formula
-------
  HH  = MAX(high, n)
  LL  = MIN(low,  n)
  %R  = (HH − close) / (HH − LL) × (−100)   — range [−100, 0]

  Rescaled to [−1, +1] for signal:
    signal = (−%R / 50) − 1      — i.e. %R=0 → +1 (oversold buy)
                                        %R=−100 → −1 (overbought sell)

Signal convention
-----------------
  +1 → %R near 0   (close at period high — overbought, SHORT signal)
  −1 → %R near −100 (close at period low  — oversold,   LONG  signal)

  Note: Williams %R logic is *inverse* to Stochastic:
    near 0 → overbought → sell; near −100 → oversold → buy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class WilliamsRIndicator(BaseIndicator):
    """Williams %R mean-reversion signal.

    Parameters
    ----------
    period     : int   — lookback window                (default 14)
    oversold   : float — %R below this → buy signal    (default −80)
    overbought : float — %R above this → sell signal   (default −20)
    """

    name = "Williams %R"
    default_params = {"period": 14, "oversold": -80.0, "overbought": -20.0}
    param_grid = {
        "period":     [7, 10, 14, 21],
        "oversold":   [-85.0, -80.0, -75.0],
        "overbought": [-25.0, -20.0, -15.0],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period:     int   = 14,
        oversold:   float = -80.0,
        overbought: float = -20.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        hh  = high.rolling(window=period, min_periods=period).max()
        ll  = low.rolling(window=period,  min_periods=period).min()
        rng = (hh - ll).replace(0.0, np.nan)

        wr  = ((hh - close) / rng * -100.0).clip(-100.0, 0.0)  # [−100, 0]

        # Map %R → signal
        # oversold  (%R ≤ −80) → +1 buy
        # overbought(%R ≥ −20) → −1 sell
        mid    = (oversold + overbought) / 2.0          # e.g. −50
        signal = pd.Series(0.0, index=wr.index)

        in_os = wr <= oversold
        signal[in_os] = ((oversold - wr[in_os]) / abs(oversold - (-100.0))).clip(0.0, 1.0)

        in_ob = wr >= overbought
        signal[in_ob] = -((wr[in_ob] - overbought) / abs(overbought)).clip(0.0, 1.0)

        mid_zone = ~in_os & ~in_ob
        half_range = abs(overbought - mid)
        if half_range > 0:
            signal[mid_zone] = -(wr[mid_zone] - mid) / half_range

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("Williams %R")
