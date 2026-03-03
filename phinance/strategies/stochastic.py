"""
phinance.strategies.stochastic
================================

Stochastic Oscillator (KDJ / %K%D) — momentum indicator.

References
----------
* Lane, G. (1957) original Stochastic Oscillator
* stockstats: kdjk / kdjd / kdjj  (KDJ variant)
* arkochhar/Technical-Indicators: STOCH()
* Stock.Indicators (.NET): Stoch()

Formula (%K / %D version)
--------------------------
  Lowest Low  (LL)  = MIN(low,  k_period)
  Highest High(HH)  = MAX(high, k_period)

  %K (fast) = 100 × (close − LL) / (HH − LL)
  %D (slow) = SMA(%K, d_period)         — "signal" line

  Signal based on %D and its distance from mid-level (50):
    signal = (50 − %D) / 50   (normalised, clipped to [−1, +1])

  Optionally smooth %K first with a 3-bar SMA (slowing):
    slow_%K = SMA(fast_%K, smooth)
    %D      = SMA(slow_%K, d_period)

Signal convention
-----------------
  +1 → %D ≤ oversold  threshold (default 20) — expect reversal up
  −1 → %D ≥ overbought threshold (default 80) — expect reversal dn
   0 → %D at 50 (neutral)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class StochasticIndicator(BaseIndicator):
    """Stochastic %D mean-reversion signal.

    Parameters
    ----------
    k_period   : int   — %K lookback window           (default 14)
    d_period   : int   — %D SMA smoothing window      (default 3)
    smooth     : int   — %K smoothing (1 = no smooth) (default 3)
    oversold   : float — %D ≤ this → +1               (default 20)
    overbought : float — %D ≥ this → −1               (default 80)
    """

    name = "Stochastic"
    default_params = {
        "k_period":   14,
        "d_period":   3,
        "smooth":     3,
        "oversold":   20.0,
        "overbought": 80.0,
    }
    param_grid = {
        "k_period":   [9, 14, 21],
        "d_period":   [3, 5],
        "smooth":     [1, 3],
        "oversold":   [15, 20, 25],
        "overbought": [75, 80, 85],
    }

    def compute(
        self,
        df: pd.DataFrame,
        k_period:   int   = 14,
        d_period:   int   = 3,
        smooth:     int   = 3,
        oversold:   float = 20.0,
        overbought: float = 80.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        ll   = low.rolling(window=k_period, min_periods=k_period).min()
        hh   = high.rolling(window=k_period, min_periods=k_period).max()
        rng  = (hh - ll).replace(0.0, np.nan)

        # Fast %K
        fast_k = ((close - ll) / rng * 100.0).clip(0.0, 100.0)

        # Slow %K (smoothed) and %D
        slow_k = fast_k.rolling(window=smooth, min_periods=1).mean() if smooth > 1 else fast_k
        pct_d  = slow_k.rolling(window=d_period, min_periods=d_period).mean()

        # Use %D (normalised, centred at 50) as a single continuous signal.
        # Formula:  signal = (50 − %D) / 50  → [-1, +1]
        #   %D = 0   → +1  (deeply oversold)
        #   %D = 50  →  0  (neutral)
        #   %D = 100 → −1  (deeply overbought)
        # NaN warmup rows are filled with 0.
        raw_signal = (50.0 - pct_d) / 50.0

        return raw_signal.fillna(0.0).clip(-1.0, 1.0).rename("Stochastic")
