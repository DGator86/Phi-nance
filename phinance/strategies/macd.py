"""
phinance.strategies.macd
========================

MACD (Moving Average Convergence/Divergence) — standard 12/26/9 EMA method.

References
----------
* Appel, G. (1979) original MACD definition
* stockstats (jealous/stockstats): macd / macds / macdh columns
* Stock.Indicators (.NET): Macd() — note: stockstats dropped the 2× scalar
  on the histogram in 2017 to match TradingView / cryptowatch convention.

Formula
-------
  MACD line      = EMA(close, fast) − EMA(close, slow)
  Signal line    = EMA(MACD line, signal)
  Histogram (h)  = MACD line − Signal line   (no extra 2× multiplier)

Signal convention
-----------------
  Histogram is normalised via 1 %/99 % quantile scaling → [−1, 1]
  Positive → bullish momentum, negative → bearish momentum.
  NaN warmup bars are filled with 0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class MACDIndicator(BaseIndicator):
    """MACD histogram momentum signal.

    Parameters
    ----------
    fast_period   : int — fast EMA window   (default 12)
    slow_period   : int — slow EMA window   (default 26)
    signal_period : int — signal EMA window (default 9)
    """

    name = "MACD"
    default_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    param_grid = {
        "fast_period":   [8, 10, 12, 16],
        "slow_period":   [21, 26, 30, 34],
        "signal_period": [7, 9, 11],
    }

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int  = 12,
        slow_period: int  = 26,
        signal_period: int = 9,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)

        ema_fast   = close.ewm(span=fast_period,   adjust=False, min_periods=fast_period).mean()
        ema_slow   = close.ewm(span=slow_period,   adjust=False, min_periods=slow_period).mean()
        macd_line  = ema_fast - ema_slow
        signal_ln  = macd_line.ewm(span=signal_period, adjust=False,
                                    min_periods=signal_period).mean()
        histogram  = macd_line - signal_ln          # no 2× multiplier (TradingView convention)

        return self._normalize_abs(histogram).fillna(0.0).clip(-1.0, 1.0).rename("MACD")
