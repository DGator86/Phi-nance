"""
phinance.strategies.mean_reversion
====================================

Mean-Reversion Z-Score indicator.

References
----------
* stockstats: close_<n>_z  (Z-score implementation)
* arkochhar/Technical-Indicators: mean-reversion via deviation

Formula
-------
  z = (close − SMA(close, n)) / σ(close, n)

  Signal = −z normalised to [−1, +1]
  Negative z (close below mean) → positive signal (expect reversal up)
  Positive z (close above mean) → negative signal (expect reversal dn)

  Clipped at ±z_threshold standard deviations before normalisation so
  extreme outliers don't destroy the scaling.

Signal convention
-----------------
  +1 → close well below rolling mean (buy mean-reversion)
  −1 → close well above rolling mean (sell mean-reversion)
   0 → close at mean
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class MeanReversionIndicator(BaseIndicator):
    """Z-score mean-reversion signal.

    Parameters
    ----------
    period      : int   — rolling window for SMA and σ (default 20)
    z_threshold : float — clip z-scores beyond ±z_threshold (default 2.0)
    """

    name = "Mean Reversion"
    default_params = {"period": 20, "z_threshold": 2.0}
    param_grid = {
        "period":      [10, 20, 30, 50],
        "z_threshold": [1.5, 2.0, 2.5],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int      = 20,
        z_threshold: float = 2.0,
        **_: Any,
    ) -> pd.Series:
        close  = df["close"].astype(float)
        sma    = close.rolling(window=period, min_periods=period).mean()
        std    = close.rolling(window=period, min_periods=period).std(ddof=1).replace(0.0, np.nan)
        z      = (close - sma) / std

        # Clip to [−z_threshold, +z_threshold] then rescale to [−1, +1]
        z_clip = z.clip(-z_threshold, z_threshold)
        signal = -(z_clip / z_threshold)   # flip sign: below mean → +signal

        return signal.fillna(0.0).clip(-1.0, 1.0).rename("Mean Reversion")
