"""
phinance.strategies.hma
========================

HMA — Hull Moving Average trend indicator.

References
----------
* Alan Hull (2005) — original HMA concept
  https://alanhull.com/hull-moving-average
* Stock.Indicators (.NET) — GetHma() series method
* TA-Lib (community fork) — HMA implementation
* https://www.fmlabs.com/reference/default.htm?url=HullMA.htm

Formula
-------
  WMA(x, period) = Weighted Moving Average with linearly increasing weights

  half_len = floor(period / 2)
  sqrt_len = round(sqrt(period))

  raw_hma = 2 × WMA(close, half_len) − WMA(close, period)
  HMA     = WMA(raw_hma, sqrt_len)

  HMA virtually eliminates lag entirely while maintaining smoothness.

Signal convention
-----------------
  Spread = (close − HMA) / HMA

  +1 → price well above HMA (uptrend)
  −1 → price well below HMA (downtrend)
   0 → neutral / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


def _wma(s: pd.Series, period: int) -> pd.Series:
    """Linearly weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)

    def _apply(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(window=period, min_periods=period).apply(_apply, raw=True)


class HMAIndicator(BaseIndicator):
    """Hull Moving Average (HMA) trend signal.

    Parameters
    ----------
    period : int — HMA window (default 20)
    """

    name = "HMA"
    default_params = {"period": 20}
    param_grid = {
        "period": [9, 14, 20, 30, 50],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)
        n = max(int(period), 4)

        half_n = max(n // 2, 2)
        sqrt_n = max(round(n ** 0.5), 2)

        wma_half = _wma(close, half_n)
        wma_full = _wma(close, n)
        raw_hma  = 2.0 * wma_half - wma_full
        hma      = _wma(raw_hma, sqrt_n)

        spread = (close - hma) / hma.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("HMA")
