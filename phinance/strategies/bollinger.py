"""
phinance.strategies.bollinger
==============================

Bollinger Bands mean-reversion indicator.

References
----------
* Bollinger, J. (2002) *Bollinger on Bollinger Bands*
* stockstats (jealous/stockstats): boll / boll_ub / boll_lb
* arkochhar/Technical-Indicators: BBAND()

Formula
-------
  Middle band = SMA(close, n)
  Upper band  = Middle + k × σ(close, n)
  Lower band  = Middle − k × σ(close, n)

  Position within bands (0 = lower, 1 = upper):
    pos = (close − lower) / (upper − lower)

  Signal = (0.5 − pos) × 2   →  +1 at lower band, −1 at upper band

Signal convention
-----------------
  +1 → close at/below lower band (mean-reversion buy)
  −1 → close at/above upper band (mean-reversion sell)
   0 → close at midband
  NaN warmup bars filled with 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class BollingerIndicator(BaseIndicator):
    """Bollinger Bands mean-reversion signal.

    Parameters
    ----------
    period  : int   — rolling SMA window (default 20)
    num_std : float — band width in σ    (default 2.0)
    """

    name = "Bollinger"
    default_params = {"period": 20, "num_std": 2.0}
    param_grid = {
        "period":  [10, 15, 20, 25, 30],
        "num_std": [1.5, 2.0, 2.5],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int   = 20,
        num_std: float = 2.0,
        **_: Any,
    ) -> pd.Series:
        close  = df["close"].astype(float)
        sma    = close.rolling(window=period, min_periods=period).mean()
        std    = close.rolling(window=period, min_periods=period).std(ddof=1)
        upper  = sma + num_std * std
        lower  = sma - num_std * std
        width  = (upper - lower).replace(0.0, np.nan)

        # Position fraction within band [0, 1]; clip so extreme outliers map to ±1
        pos    = ((close - lower) / width).clip(0.0, 1.0)
        signal = (0.5 - pos) * 2.0          # lower → +1, upper → −1

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("Bollinger")
