"""
phinance.strategies.kama
=========================

KAMA — Kaufman Adaptive Moving Average trend indicator.

References
----------
* Perry J. Kaufman (1998) — "Trading Systems and Methods", 3rd ed.
* Stock.Indicators (.NET) — GetKama() series method
* arkochhar/Technical-Indicators — KAMA implementation
* https://www.investopedia.com/terms/k/kaufman_adaptive_moving_average.asp

Formula
-------
  Efficiency Ratio (ER):
    direction = abs(close[t] − close[t-er_period])
    volatility = sum(abs(close[i] − close[i-1]), i=t-er_period+1..t)
    ER = direction / volatility  (0 when volatility=0)

  Smoothing Constant (SC):
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    SC = (ER × (fast_sc − slow_sc) + slow_sc) ^ 2

  KAMA(t) = KAMA(t-1) + SC × (close[t] − KAMA(t-1))
  Seed: KAMA[er_period] = close[er_period]

Signal convention
-----------------
  Spread = (close − KAMA) / KAMA

  +1 → price well above KAMA (strong trend)
  −1 → price well below KAMA (strong downtrend)
   0 → neutral / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class KAMAIndicator(BaseIndicator):
    """Kaufman Adaptive Moving Average (KAMA) trend signal.

    Parameters
    ----------
    er_period  : int — Efficiency Ratio period (default 10)
    fast_period: int — fast EMA period when trending (default 2)
    slow_period: int — slow EMA period when ranging (default 30)
    """

    name = "KAMA"
    default_params = {"er_period": 10, "fast_period": 2, "slow_period": 30}
    param_grid = {
        "er_period":   [5, 8, 10, 14, 20],
        "fast_period": [2, 3, 5],
        "slow_period": [20, 30, 40],
    }

    def compute(
        self,
        df: pd.DataFrame,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float).values
        n = len(close)
        er_n = max(int(er_period), 2)
        fast_sc = 2.0 / (max(int(fast_period), 2) + 1)
        slow_sc = 2.0 / (max(int(slow_period), 2) + 1)

        kama = np.full(n, np.nan)
        if er_n >= n:
            return pd.Series(0.0, index=df.index, name="KAMA")

        # Seed KAMA at the first valid ER bar
        kama[er_n] = close[er_n]

        for i in range(er_n + 1, n):
            direction  = abs(close[i] - close[i - er_n])
            volatility = np.sum(np.abs(np.diff(close[i - er_n: i + 1])))
            er = direction / volatility if volatility > 0 else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

        kama_s = pd.Series(kama, index=df.index)
        close_s = pd.Series(close, index=df.index)
        spread  = (close_s - kama_s) / kama_s.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("KAMA")
