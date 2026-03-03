"""
phinance.strategies.ichimoku
=============================

Ichimoku Kinko Hyo — Japanese equilibrium chart trend & momentum indicator.

References
----------
* Goichi Hosoda (1969) — original Ichimoku Kinko Hyo system
* Stock.Indicators (.NET) — GetIchimoku() series method
* https://www.investopedia.com/terms/i/ichimoku-cloud.asp
* https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ichimoku-cloud

Formula (standard parameters)
------------------------------
  Tenkan-sen  (Conversion, fast_period=9):
    (highest_high + lowest_low) / 2  over fast_period bars

  Kijun-sen   (Base line, slow_period=26):
    (highest_high + lowest_low) / 2  over slow_period bars

  Senkou Span A (Leading Span A, plotted cloud_period bars ahead):
    (Tenkan + Kijun) / 2  shifted forward cloud_period bars

  Senkou Span B (Leading Span B, plotted cloud_period bars ahead):
    (highest_high + lowest_low) / 2  over (2 × slow_period) bars,
    shifted forward cloud_period bars

  Chikou Span (Lagging):
    close shifted back cloud_period bars

  Cloud (Kumo) = region between Span A and Span B

Signal convention
-----------------
  For current-bar signal we use un-shifted Span A / Span B (contemporaneous).

  Score components (each ±1 or 0):
    1. Price above Tenkan  (+0.4 / −0.4)
    2. Price above Kijun   (+0.4 / −0.4)
    3. Tenkan above Kijun  (+0.2 / −0.2)

  Signal = clamp(sum of components, −1, 1)

  +1 → price above both lines, Tenkan above Kijun → strong uptrend
  −1 → price below both lines, Tenkan below Kijun → strong downtrend
   0 → conflicting / warmup
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class IchimokuIndicator(BaseIndicator):
    """Ichimoku Kinko Hyo trend/momentum signal.

    Parameters
    ----------
    fast_period  : int — Tenkan-sen period (default 9)
    slow_period  : int — Kijun-sen period (default 26)
    cloud_period : int — Senkou displacement period (default 26)
    """

    name = "Ichimoku"
    default_params = {"fast_period": 9, "slow_period": 26, "cloud_period": 26}
    param_grid = {
        "fast_period":  [7, 9, 12],
        "slow_period":  [20, 26, 34],
        "cloud_period": [26, 52],
    }

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int = 9,
        slow_period: int = 26,
        cloud_period: int = 26,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        fn = max(int(fast_period), 2)
        sn = max(int(slow_period), 2)

        # Tenkan-sen and Kijun-sen (mid-point of range)
        tenkan = (
            high.rolling(fn, min_periods=fn).max()
            + low.rolling(fn, min_periods=fn).min()
        ) / 2.0

        kijun = (
            high.rolling(sn, min_periods=sn).max()
            + low.rolling(sn, min_periods=sn).min()
        ) / 2.0

        # Composite signal from three binary votes
        above_tenkan = np.where(close > tenkan,  0.4, np.where(close < tenkan, -0.4, 0.0))
        above_kijun  = np.where(close > kijun,   0.4, np.where(close < kijun,  -0.4, 0.0))
        tk_cross     = np.where(tenkan > kijun,  0.2, np.where(tenkan < kijun, -0.2, 0.0))

        # Replace NaN positions with 0
        raw = pd.Series(above_tenkan + above_kijun + tk_cross, index=df.index)
        nan_mask = tenkan.isna() | kijun.isna()
        raw[nan_mask] = 0.0

        return raw.clip(-1.0, 1.0).fillna(0.0).rename("Ichimoku")
