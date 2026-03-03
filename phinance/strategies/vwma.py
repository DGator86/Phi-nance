"""
phinance.strategies.vwma
=========================

VWMA — Volume Weighted Moving Average trend indicator.

References
----------
* Stock.Indicators (.NET) — GetVwma() series method
* TradingView — built-in vwma()
* https://www.investopedia.com/terms/v/volume-weighted-moving-average.asp

Formula
-------
  VWMA(period) = sum(close[i] × volume[i], i=t-period+1..t)
                 / sum(volume[i], i=t-period+1..t)

  VWMA is a price-weighted average that gives more influence to bars with
  higher volume.  It diverges from a plain MA when volume is concentrated
  at specific price levels (e.g., breakout bars).

Signal convention
-----------------
  Spread = (close − VWMA) / VWMA

  +1 → price well above volume-weighted average (bullish)
  −1 → price well below volume-weighted average (bearish)
   0 → neutral / warmup / no volume

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class VWMAIndicator(BaseIndicator):
    """Volume Weighted Moving Average (VWMA) trend signal.

    Parameters
    ----------
    period : int — rolling window in bars (default 20)
    """

    name = "VWMA"
    default_params = {"period": 20}
    param_grid = {
        "period": [10, 14, 20, 30, 50],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        **_: Any,
    ) -> pd.Series:
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)
        n = max(int(period), 2)

        pv     = close * volume
        vwma   = (
            pv.rolling(n, min_periods=n).sum()
            / volume.rolling(n, min_periods=n).sum()
        )

        spread = (close - vwma) / vwma.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("VWMA")
