"""
phinance.strategies.ema
========================

EMA Cross (Exponential Moving Average cross-over) indicator.

References
----------
* stockstats: close_<n>_ema
* arkochhar/Technical-Indicators: EMA()
* Stock.Indicators (.NET): Ema()

Formula
-------
  ema_fast = EMA(close, fast_period)   — span = fast_period
  ema_slow = EMA(close, slow_period)   — span = slow_period
  spread   = (ema_fast − ema_slow) / ema_slow

  Normalised via 1%/99% quantile scaling → [−1, +1].

Signal convention
-----------------
  +1 → fast EMA well above slow EMA (strong uptrend)
  −1 → fast EMA well below slow EMA (strong downtrend)
   0 → EMAs at parity / warmup

  Unlike Dual SMA, EMA reacts faster and is more sensitive to recent prices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class EMACrossIndicator(BaseIndicator):
    """EMA cross-over trend-following signal.

    Parameters
    ----------
    fast_period : int — fast EMA window (default 12)
    slow_period : int — slow EMA window (default 26)
    """

    name = "EMA Cross"
    default_params = {"fast_period": 12, "slow_period": 26}
    param_grid = {
        "fast_period": [5, 8, 12, 20],
        "slow_period": [20, 26, 34, 50],
    }

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        **_: Any,
    ) -> pd.Series:
        close    = df["close"].astype(float)
        ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
        spread   = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)

        return self._normalize(spread).fillna(0.0).clip(-1.0, 1.0).rename("EMA Cross")
