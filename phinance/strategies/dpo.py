"""
phinance.strategies.dpo
========================

DPO — Detrended Price Oscillator cycle/rhythm indicator.

References
----------
* Stock.Indicators (.NET) — GetDpo() series method
* https://www.investopedia.com/terms/d/detrended-price-oscillator.asp
* https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/detrended-price-oscillator-dpo

Formula
-------
  lookback = floor(period / 2) + 1
  SMA      = rolling_mean(close, period)

  DPO(t) = close[t − lookback] − SMA[t]

  By subtracting the centered SMA from a past close, DPO removes the trend
  component and isolates shorter-term cycle oscillations.

Signal convention
-----------------
  Positive DPO → cycle at a peak (potential overbought / mean-reversion sell)
  Negative DPO → cycle at a trough (potential oversold / mean-reversion buy)

  Because DPO detects cycles rather than trends, the signal polarity is
  contrarian:
    +1 → DPO strongly negative (oversold cycle trough → buy)
    −1 → DPO strongly positive (overbought cycle peak → sell)
     0 → DPO near zero / warmup

  The raw DPO is normalised via sign-preserving _normalize_abs and then
  negated so that negative DPO (cycle low) maps to a positive (buy) signal.

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class DPOIndicator(BaseIndicator):
    """Detrended Price Oscillator (DPO) cycle signal.

    Parameters
    ----------
    period : int — SMA period (default 20)
    """

    name = "DPO"
    default_params = {"period": 20}
    param_grid = {
        "period": [10, 14, 20, 30, 40],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)
        n = max(int(period), 2)
        lookback = n // 2 + 1

        sma = close.rolling(n, min_periods=n).mean()
        dpo = close.shift(lookback) - sma

        # Contrarian: negative DPO → buy (+), positive DPO → sell (−)
        signal = -self._normalize_abs(dpo)
        return signal.fillna(0.0).clip(-1.0, 1.0).rename("DPO")
