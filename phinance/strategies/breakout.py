"""
phinance.strategies.breakout
==============================

Donchian Channel breakout indicator.

References
----------
* Donchian, R.D. (1970s) original channel breakout rules
* arkochhar/Technical-Indicators — channel breakout logic
* Stock.Indicators (.NET): DonchianChannels()

Formula
-------
  upper_ch = MAX(high, n)       # Donchian upper channel
  lower_ch = MIN(low,  n)       # Donchian lower channel
  mid_ch   = (upper_ch + lower_ch) / 2

  position = (close − lower_ch) / (upper_ch − lower_ch)  → [0, 1]
  signal   = (position − 0.5) × 2                        → [−1, +1]
    +1 → close at top of channel (breakout up)
    −1 → close at bottom of channel (breakout dn)
     0 → close at midpoint

Signal convention
-----------------
  +1 → price at top of Donchian channel (momentum breakout signal)
  −1 → price at bottom of channel
   0 → price in middle / warmup
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class BreakoutIndicator(BaseIndicator):
    """Donchian channel breakout signal.

    Parameters
    ----------
    period : int — channel lookback window (default 20)
    """

    name = "Breakout"
    default_params = {"period": 20}
    param_grid = {
        "period": [10, 15, 20, 30, 50],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        **_: Any,
    ) -> pd.Series:
        high     = df["high"].astype(float)
        low      = df["low"].astype(float)
        close    = df["close"].astype(float)

        upper_ch = high.rolling(window=period, min_periods=period).max()
        lower_ch = low.rolling(window=period, min_periods=period).min()
        width    = (upper_ch - lower_ch).replace(0.0, np.nan)

        position = ((close - lower_ch) / width).clip(0.0, 1.0)
        signal   = (position - 0.5) * 2.0

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("Breakout")
