"""
phinance.strategies.keltner
============================

Keltner Channel — volatility-based trend / breakout indicator.

References
----------
* Chester Keltner (1960) — original Keltner Channel concept
* Linda Bradford Raschke — modern ATR-based formulation
* Stock.Indicators (.NET) — GetKeltner() series method
* https://www.investopedia.com/terms/k/keltnerchannel.asp

Formula (modern ATR-based version)
------------------------------------
  middle = EMA(close, period)
  ATR    = RMA(true_range, period)   (Wilder's smoothing = EMA span=2*period-1)
  upper  = middle + multiplier × ATR
  lower  = middle − multiplier × ATR

  True range = max(high − low,
                   abs(high − prev_close),
                   abs(low  − prev_close))

  Position = (close − middle) / (multiplier × ATR)

  Positive → close above EMA relative to channel width (bullish)
  Negative → close below EMA relative to channel width (bearish)

Signal convention
-----------------
  +1 → close at / above upper band (breakout up)
  −1 → close at / below lower band (breakout down)
   0 → close near EMA / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class KeltnerIndicator(BaseIndicator):
    """Keltner Channel breakout / volatility signal.

    Parameters
    ----------
    period     : int   — EMA and ATR period (default 20)
    multiplier : float — ATR band multiplier (default 2.0)
    """

    name = "Keltner"
    default_params = {"period": 20, "multiplier": 2.0}
    param_grid = {
        "period":     [10, 14, 20, 30],
        "multiplier": [1.5, 2.0, 2.5],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 20,
        multiplier: float = 2.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        n = max(int(period), 2)
        mult = float(multiplier)

        # True range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        # Wilder's smoothing for ATR: equivalent to EMA with span = 2*n - 1
        atr    = tr.ewm(span=2 * n - 1, adjust=False, min_periods=n).mean()
        middle = close.ewm(span=n, adjust=False, min_periods=n).mean()
        band   = mult * atr

        position = (close - middle) / band.replace(0.0, np.nan)
        return position.clip(-1.0, 1.0).fillna(0.0).rename("Keltner")
