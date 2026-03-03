"""
phinance.strategies.dema
=========================

DEMA — Double Exponential Moving Average trend indicator.

References
----------
* Patrick Mulloy (1994) — original DEMA paper, *Technical Analysis of Stocks
  & Commodities*
* Stock.Indicators (.NET) — GetDema() series method
* TA-Lib — DEMA implementation
* https://www.investopedia.com/articles/trading/10/double-exponential-moving-average.asp

Formula
-------
  EMA1 = EMA(close, period)
  EMA2 = EMA(EMA1, period)
  DEMA = 2 × EMA1 − EMA2

  DEMA reacts faster than a simple EMA because the double-smoothed component
  is subtracted, reducing lag by approximately half.

Signal convention
-----------------
  Spread = (close − DEMA) / DEMA

  +1 → price well above DEMA (strong uptrend)
  −1 → price well below DEMA (strong downtrend)
   0 → price at DEMA / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class DEMAIndicator(BaseIndicator):
    """Double Exponential Moving Average (DEMA) trend signal.

    Parameters
    ----------
    period : int — EMA window (default 21)
    """

    name = "DEMA"
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

        ema1 = close.ewm(span=n, adjust=False, min_periods=n).mean()
        ema2 = ema1.ewm(span=n, adjust=False, min_periods=n).mean()
        dema = 2.0 * ema1 - ema2

        spread = (close - dema) / dema.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("DEMA")
