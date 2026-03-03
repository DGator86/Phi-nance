"""
phinance.strategies.zlema
==========================

ZLEMA — Zero Lag Exponential Moving Average trend indicator.

References
----------
* John Ehlers & Ric Way (2010) — "Zero Lag (Well, Almost)"
  *Stocks & Commodities*, Jan 2010
* Stock.Indicators (.NET) — GetZlEma() series method
* https://www.investopedia.com/terms/z/zero-lag-exponential-moving-average.asp

Formula
-------
  lag   = floor((period − 1) / 2)
  ema_src = 2 × close − close.shift(lag)      ← lag-corrected input
  ZLEMA = EMA(ema_src, period)

  The lag correction pre-shifts the price series to cancel EMA's inherent
  lag, resulting in near-zero delay while retaining EMA smoothing.

Signal convention
-----------------
  Spread = (close − ZLEMA) / ZLEMA

  +1 → price well above ZLEMA (uptrend)
  −1 → price well below ZLEMA (downtrend)
   0 → neutral / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class ZLEMAIndicator(BaseIndicator):
    """Zero Lag EMA (ZLEMA) trend signal.

    Parameters
    ----------
    period : int — EMA window (default 21)
    """

    name = "ZLEMA"
    default_params = {"period": 21}
    param_grid = {
        "period": [10, 14, 21, 34, 55],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 21,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)
        n = max(int(period), 2)
        lag = (n - 1) // 2

        ema_src = 2.0 * close - close.shift(lag)
        zlema   = ema_src.ewm(span=n, adjust=False, min_periods=n).mean()

        spread = (close - zlema) / zlema.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("ZLEMA")
