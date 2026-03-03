"""
phinance.strategies.donchian
=============================

Donchian Channel — breakout / trend-following indicator.

References
----------
* Richard Donchian (1960s) — original Donchian Channel concept
* Stock.Indicators (.NET) — GetDonchian() series method
* TradingView — built-in dc()
* https://www.investopedia.com/terms/d/donchianchannels.asp

Formula
-------
  upper  = max(high, period)
  lower  = min(low,  period)
  middle = (upper + lower) / 2
  width  = upper − lower

  Position = (close − middle) / (width / 2)   ∈ [−1, +1]

  Positive → close in upper half of channel (bullish momentum)
  Negative → close in lower half of channel (bearish momentum)
  Near ±1  → breakout signal

Signal convention
-----------------
  The channel position is already in [−1, +1]; used directly.

  +1 → close at the top of the Donchian channel (strong breakout up)
  −1 → close at the bottom of the Donchian channel (strong breakout down)
   0 → close at channel midpoint / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class DonchianIndicator(BaseIndicator):
    """Donchian Channel position signal.

    Parameters
    ----------
    period : int — rolling window for highest-high / lowest-low (default 20)
    """

    name = "Donchian"
    default_params = {"period": 20}
    param_grid = {
        "period": [10, 14, 20, 30, 55],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        n = max(int(period), 2)

        upper  = high.rolling(n, min_periods=n).max()
        lower  = low.rolling(n, min_periods=n).min()
        middle = (upper + lower) / 2.0
        half_w = (upper - lower) / 2.0

        # Position in [−1, +1]; 0 where half_w == 0 or NaN
        position = (close - middle) / half_w.replace(0.0, np.nan)
        return position.clip(-1.0, 1.0).fillna(0.0).rename("Donchian")
