"""
phinance.strategies.elder_ray
==============================

Elder Ray Index — trend-following indicator measuring bull/bear power.

References
----------
* Dr. Alexander Elder (1993) — "Trading for a Living"
* Stock.Indicators (.NET) — GetElderRay() series method
* https://www.investopedia.com/terms/e/elderray.asp

Formula
-------
  EMA    = EMA(close, period)
  Bull Power = high − EMA     (how far bulls pushed price above EMA)
  Bear Power = low  − EMA     (how far bears pushed price below EMA)

  Net Power = Bull Power + Bear Power
            = (high + low) − 2 × EMA
            = 2 × (midpoint − EMA)

  The combined Net Power is normalised to [−1, +1].

Signal convention
-----------------
  +1 → strong bull dominance (high and low both well above EMA)
  −1 → strong bear dominance (high and low both well below EMA)
   0 → EMA at midpoint / warmup

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class ElderRayIndicator(BaseIndicator):
    """Elder Ray Index combined net-power signal.

    Parameters
    ----------
    period : int — EMA period (default 13)
    """

    name = "Elder Ray"
    default_params = {"period": 13}
    param_grid = {
        "period": [8, 10, 13, 20, 26],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 13,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        n = max(int(period), 2)

        ema        = close.ewm(span=n, adjust=False, min_periods=n).mean()
        bull_power = high - ema
        bear_power = low  - ema
        net_power  = bull_power + bear_power   # = (high + low) − 2 × EMA

        return self._normalize_abs(net_power).fillna(0.0).clip(-1.0, 1.0).rename("Elder Ray")
