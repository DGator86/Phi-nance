"""
phinance.strategies.tema
=========================

TEMA — Triple Exponential Moving Average trend indicator.

References
----------
* Patrick Mulloy (1994) — extended TEMA concept
* Stock.Indicators (.NET) — GetTema() series method
* TA-Lib — TEMA implementation
* https://www.investopedia.com/articles/trading/10/double-exponential-moving-average.asp

Formula
-------
  EMA1 = EMA(close, period)
  EMA2 = EMA(EMA1, period)
  EMA3 = EMA(EMA2, period)
  TEMA = 3 × EMA1 − 3 × EMA2 + EMA3

  TEMA further reduces lag relative to DEMA by applying a third-order
  correction, making it highly responsive to recent price changes.

Signal convention
-----------------
  Spread = (close − TEMA) / TEMA

  +1 → price well above TEMA (strong uptrend)
  −1 → price well below TEMA (strong downtrend)
   0 → price at TEMA / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class TEMAIndicator(BaseIndicator):
    """Triple Exponential Moving Average (TEMA) trend signal.

    Parameters
    ----------
    period : int — EMA window (default 21)
    """

    name = "TEMA"
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
        ema3 = ema2.ewm(span=n, adjust=False, min_periods=n).mean()
        tema = 3.0 * ema1 - 3.0 * ema2 + ema3

        spread = (close - tema) / tema.replace(0.0, np.nan)
        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("TEMA")
