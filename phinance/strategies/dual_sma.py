"""
phinance.strategies.dual_sma
=============================

Dual Simple Moving Average cross-over trend-following indicator.

References
----------
* stockstats: close_<n>_sma / cross detection pattern
* arkochhar/Technical-Indicators: SMA cross logic

Formula
-------
  fast_sma  = SMA(close, fast_period)
  slow_sma  = SMA(close, slow_period)
  spread    = (fast_sma − slow_sma) / slow_sma   (normalised %)

  Signal is normalised via 1 %/99 % quantile scaling → [−1, 1].

Signal convention
-----------------
  +1 → fast SMA well above slow SMA (strong uptrend)
  −1 → fast SMA well below slow SMA (strong downtrend)
   0 → SMAs at parity / warmup
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class DualSMAIndicator(BaseIndicator):
    """Dual SMA trend-following signal.

    Parameters
    ----------
    fast_period : int — fast SMA window  (default 10)
    slow_period : int — slow SMA window  (default 50)
    """

    name = "Dual SMA"
    default_params = {"fast_period": 10, "slow_period": 50}
    param_grid = {
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100, 200],
    }

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 50,
        **_: Any,
    ) -> pd.Series:
        close    = df["close"].astype(float)
        sma_fast = close.rolling(window=fast_period, min_periods=fast_period).mean()
        sma_slow = close.rolling(window=slow_period, min_periods=slow_period).mean()
        spread   = (sma_fast - sma_slow) / sma_slow.replace(0.0, np.nan)

        return self._normalize_abs(spread).fillna(0.0).clip(-1.0, 1.0).rename("Dual SMA")
