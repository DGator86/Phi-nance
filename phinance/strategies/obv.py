"""
phinance.strategies.obv
========================

OBV (On-Balance Volume) momentum indicator.

References
----------
* Granville, J. (1963) original OBV concept
* stockstats: close/volume delta logic
* Finance-Python (alpha-miner): volume-based momentum

Formula
-------
  OBV[0] = 0
  OBV[t] = OBV[t-1] + volume[t]   if close[t] > close[t-1]
          = OBV[t-1] − volume[t]   if close[t] < close[t-1]
          = OBV[t-1]               if close[t] == close[t-1]

  Signal = ROC(OBV, n)      (rate of change of OBV over n bars)
  Then normalised via 1%/99% quantile scaling → [−1, +1].

Signal convention
-----------------
  +1 → OBV strongly rising (volume confirming upward price move)
  −1 → OBV strongly falling (volume confirming downward price move)
   0 → OBV flat / warmup
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class OBVIndicator(BaseIndicator):
    """OBV rate-of-change momentum signal.

    Parameters
    ----------
    period : int — ROC window for OBV slope (default 14)
    """

    name = "OBV"
    default_params = {"period": 14}
    param_grid = {
        "period": [7, 10, 14, 21, 30],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        **_: Any,
    ) -> pd.Series:
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        direction = np.sign(close.diff().fillna(0))
        obv       = (direction * volume).cumsum()

        # Rate-of-change over `period` bars
        obv_prev = obv.shift(period)
        roc      = (obv - obv_prev) / (obv_prev.abs().replace(0.0, np.nan))

        return self._normalize_abs(roc).fillna(0.0).clip(-1.0, 1.0).rename("OBV")
