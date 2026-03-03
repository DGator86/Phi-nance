"""
phinance.strategies.mass_index
================================

Mass Index — trend-reversal indicator based on high-low range expansion.

References
----------
* Dorsey, D. (1992) — original Mass Index article, *Technical Analysis of
  Stocks & Commodities*
* Stock.Indicators (.NET) — GetMassIndex() series method
* stockstats — mass_index implementation
* arkochhar/Technical-Indicators — MassIndex()
* https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/mass-index
* https://en.wikipedia.org/wiki/Mass_index

Formula
-------
  Single EMA  = EMA(high − low, fast_period)
  Double EMA  = EMA(Single EMA, fast_period)

  EMA Ratio   = Single EMA / Double EMA

  Mass Index  = SUM(EMA Ratio, slow_period)        [rolling sum over slow bars]

  Classic parameters: fast_period = 9, slow_period = 25.

  The "Reversal Bulge" signal:
    • When Mass Index rises above 27 → bulge forming (range expansion)
    • When Mass Index then falls below 26.5 → reversal bulge confirmed
      → expect trend reversal (direction determined by other indicators)

  For the Phi-nance platform we generate a *mean-reversion* signal:
    • MI above bulge_high (default 27) maps to +0.5 (range expansion, watch
      for reversal — slightly bullish if combined with downtrend)
    • We actually model it as: signal = normalize(MI − midpoint) then
      use sign flip so that extreme MI (potential reversal) gives a non-zero
      signal. The simplest consistent approach:

    raw = (MI − mean_MI) / std_MI    ← z-score
    signal = raw.clip(-1, 1) with sign flipped so extreme = reversal bias

  Because Mass Index is a *reversal* indicator (extremes = expected reversal),
  we flip the sign of the z-score so that:
    very high MI → negative signal (over-extended range, reversal expected)
    very low  MI → positive signal (compressed range, expansion / continuation)

Signal convention
-----------------
  +1 → MI very low (range compressed — possible continuation or early breakout)
  −1 → MI very high (reversal bulge — range expansion, reversal expected)
   0 → MI at historical median
  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class MassIndexIndicator(BaseIndicator):
    """Mass Index trend-reversal signal.

    Parameters
    ----------
    fast_period : int   — EMA period for high-low range (default 9)
    slow_period : int   — rolling sum period for EMA ratio (default 25)
    bulge_high  : float — reversal bulge upper threshold (default 27.0)
    bulge_low   : float — reversal bulge trigger threshold (default 26.5)

    Notes
    -----
    The Dorsey original uses fast=9, slow=25.  The signal is inverted so
    that high Mass Index (potential reversal) maps to −1 signal, while a
    low MI maps to +1 (calm, range compressed).
    """

    name = "Mass Index"
    default_params = {
        "fast_period": 9,
        "slow_period": 25,
        "bulge_high":  27.0,
        "bulge_low":   26.5,
    }
    param_grid = {
        "fast_period": [7, 9, 12],
        "slow_period": [20, 25, 30],
    }

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int   = 9,
        slow_period: int   = 25,
        bulge_high:  float = 27.0,
        bulge_low:   float = 26.5,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        fast = int(fast_period)
        slow = int(slow_period)
        if fast < 1:
            fast = 1
        if slow < 1:
            slow = 1

        # High-low range
        hl_range = high - low

        # Single EMA of range
        single_ema = hl_range.ewm(span=fast, min_periods=fast, adjust=False).mean()

        # Double EMA of single EMA
        double_ema = single_ema.ewm(span=fast, min_periods=fast, adjust=False).mean()

        # EMA ratio (avoid division by zero)
        ema_ratio = single_ema / double_ema.where(double_ema != 0, other=1e-10)

        # Mass Index = rolling sum of EMA ratios
        mass_index = ema_ratio.rolling(window=slow, min_periods=slow).sum()

        # Z-score and invert: high MI (reversal) → negative signal
        mi_mean = mass_index.rolling(window=slow * 2, min_periods=slow).mean()
        mi_std  = mass_index.rolling(window=slow * 2, min_periods=slow).std()

        z_score = (mass_index - mi_mean) / mi_std.where(mi_std != 0, other=1e-10)

        # Flip sign: high MI (risk of reversal) maps to negative signal
        signal = (-z_score).clip(-1.0, 1.0)

        return signal.fillna(0.0).rename("Mass Index")
