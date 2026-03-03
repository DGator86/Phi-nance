"""
phinance.strategies.vwap
=========================

VWAP (Volume-Weighted Average Price) deviation indicator.

References
----------
* stockstats: vwma (volume-weighted MA variant)
* arkochhar/Technical-Indicators: VWAP
* Standard institutional VWAP formula

Formula
-------
  typical_price = (high + low + close) / 3
  vwap          = cumsum(typical_price × volume) / cumsum(volume)
    — reset daily (or use rolling window for multi-day data)

  For multi-day / daily OHLCV data a rolling-window VWAP is used:
    rolling_vwap[t] = sum(tp × vol, n) / sum(vol, n)

  deviation = (close − rolling_vwap) / rolling_vwap × 100  (%)

  Signal clipped at ±band_pct % then scaled to [−1, +1]:
    +1 → close > vwap + band_pct %  (overbought relative to VWAP)
    −1 → close < vwap − band_pct %  (oversold relative to VWAP)
     0 → close at VWAP

Signal convention
-----------------
  The VWAP signal is *contrarian* (mean-reverting):
  price above VWAP → sell signal (−1), below VWAP → buy signal (+1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class VWAPIndicator(BaseIndicator):
    """Rolling VWAP deviation mean-reversion signal.

    Parameters
    ----------
    period   : int   — rolling VWAP window in bars (default 20)
    band_pct : float — deviation threshold in % (default 0.5)
    """

    name = "VWAP"
    default_params = {"period": 20, "band_pct": 0.5}
    param_grid = {
        "period":   [10, 20, 30, 50],
        "band_pct": [0.25, 0.5, 1.0, 1.5],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int    = 20,
        band_pct: float = 0.5,
        **_: Any,
    ) -> pd.Series:
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float).replace(0.0, np.nan)

        tp  = (high + low + close) / 3.0
        pv  = tp * volume

        sum_pv  = pv.rolling(window=period, min_periods=1).sum()
        sum_vol = volume.rolling(window=period, min_periods=1).sum().replace(0.0, np.nan)
        vwap    = sum_pv / sum_vol

        # Deviation from VWAP in %
        dev     = (close - vwap) / vwap.replace(0.0, np.nan) * 100.0

        # Clip to ±band_pct and normalise to [−1, +1]; flip sign (contrarian)
        if band_pct == 0:
            band_pct = 0.01
        signal  = -(dev / band_pct).clip(-1.0, 1.0)

        return signal.fillna(0.0).rename("VWAP")
