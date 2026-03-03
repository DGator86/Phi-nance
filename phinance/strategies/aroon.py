"""
phinance.strategies.aroon
==========================

Aroon Indicator — trend strength and direction oscillator.

References
----------
* Tushar Chande (1995) — original Aroon concept
* Stock.Indicators (.NET) — GetAroon() series method
* stockstats — aroon_osc implementation
* arkochhar/Technical-Indicators — Aroon implementation
* https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon-indicator
* https://wire.insiderfinance.io/how-to-calculate-aroon-indicator-in-python-7ca9900107cd

Formula
-------
  Aroon Up   = ((period − days_since_period_high) / period) × 100
  Aroon Down = ((period − days_since_period_low)  / period) × 100

  Aroon Oscillator = Aroon Up − Aroon Down  ∈ [−100, +100]

  ``days_since_period_high`` is the number of bars since the highest high
  within the rolling ``period``-bar window (0 means the high was today).
  Equivalently: ``period − argmax(high[-period:])``

Signal convention
-----------------
  The Aroon Oscillator is normalised to [−1, +1] by dividing by 100.

  +1 → Aroon Up = 100, Aroon Down = 0 → strong uptrend
  −1 → Aroon Up = 0,   Aroon Down = 100 → strong downtrend
   0 → Aroon Up ≈ Aroon Down → trendless / consolidating

  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class AroonIndicator(BaseIndicator):
    """Aroon trend-strength signal using the Aroon Oscillator.

    Parameters
    ----------
    period : int — lookback window for highest-high / lowest-low (default 25)

    Notes
    -----
    The classic Aroon uses period = 25 (5 trading weeks).
    Shorter periods (10–14) are common for swing traders.
    """

    name = "Aroon"
    default_params = {"period": 25}
    param_grid = {
        "period": [10, 14, 20, 25, 30],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 25,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        n = int(period)
        if n < 1:
            n = 1

        # rolling argmax / argmin over the window of size (n+1) —
        # include today plus the previous n bars so the window has n+1 points
        window = n + 1

        def _days_since_high(x: np.ndarray) -> float:
            """Bars since highest high in window (0 = current bar is the high)."""
            # argmax returns index of max; last element is current bar
            return float(len(x) - 1 - np.argmax(x))

        def _days_since_low(x: np.ndarray) -> float:
            """Bars since lowest low in window."""
            return float(len(x) - 1 - np.argmin(x))

        days_h = high.rolling(window=window, min_periods=window).apply(
            _days_since_high, raw=True
        )
        days_l = low.rolling(window=window, min_periods=window).apply(
            _days_since_low, raw=True
        )

        aroon_up   = ((n - days_h) / n) * 100.0
        aroon_down = ((n - days_l) / n) * 100.0

        # Oscillator ∈ [−100, +100]
        oscillator = aroon_up - aroon_down

        # Normalise to [−1, +1]
        signal = (oscillator / 100.0).clip(-1.0, 1.0)

        return signal.fillna(0.0).rename("Aroon")
