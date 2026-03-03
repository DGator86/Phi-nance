"""
phinance.strategies.psar
=========================

Parabolic SAR (Stop And Reverse) trend indicator.

References
----------
* Wilder, J.W. (1978) *New Concepts in Technical Trading Systems* — Ch. 8
* arkochhar/Technical-Indicators: SAR()
* Stock.Indicators (.NET): ParabolicSar()

Formula (Wilder's iterative algorithm)
---------------------------------------
  State variables: sar (stop-and-reverse price), ep (extreme point),
                   af (acceleration factor), long (bool: currently long)

  Initial state: long = True, sar = first low, ep = first high, af = initial_af

  Each bar:
    if long:
      sar_new = sar + af × (ep − sar)
      sar_new = min(sar_new, low[t-1], low[t-2])
      if high[t] > ep:  ep = high[t]; af = min(af + step_af, max_af)
      if low[t] < sar_new:  flip to short
    else:
      sar_new = sar + af × (ep − sar)
      sar_new = max(sar_new, high[t-1], high[t-2])
      if low[t] < ep:  ep = low[t]; af = min(af + step_af, max_af)
      if high[t] > sar_new:  flip to long

  Signal: (close − SAR) / SAR × 100  — normalised price distance from SAR

Signal convention
-----------------
  +1 → price well above SAR (strong uptrend; SAR acts as trailing stop below)
  −1 → price well below SAR (strong downtrend; SAR acts as trailing stop above)
   0 → price near SAR / transition zone
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class PSARIndicator(BaseIndicator):
    """Parabolic SAR trend-following signal.

    Parameters
    ----------
    initial_af : float — starting acceleration factor (default 0.02)
    step_af    : float — AF increment per new extreme (default 0.02)
    max_af     : float — maximum acceleration factor  (default 0.20)
    scale      : float — normalisation: %distance clipped at ±scale (default 2.0)
    """

    name = "PSAR"
    default_params = {
        "initial_af": 0.02,
        "step_af":    0.02,
        "max_af":     0.20,
        "scale":      2.0,
    }
    param_grid = {
        "initial_af": [0.01, 0.02, 0.03],
        "step_af":    [0.01, 0.02, 0.03],
        "max_af":     [0.10, 0.20, 0.30],
    }

    def compute(
        self,
        df: pd.DataFrame,
        initial_af: float = 0.02,
        step_af:    float = 0.02,
        max_af:     float = 0.20,
        scale:      float = 2.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float).values
        low   = df["low"].astype(float).values
        close = df["close"].astype(float).values
        n     = len(close)

        sar_arr = np.zeros(n)
        # We need at least 2 bars
        if n < 3:
            return pd.Series(0.0, index=df.index).rename("PSAR")

        # Seed from bar 0
        bull = close[1] >= close[0]   # initial trend guess
        sar  = low[0]  if bull else high[0]
        ep   = high[0] if bull else low[0]
        af   = initial_af

        sar_arr[0] = sar
        sar_arr[1] = sar

        for i in range(2, n):
            prev_sar = sar
            # Update SAR
            sar = prev_sar + af * (ep - prev_sar)

            if bull:
                sar = min(sar, low[i - 1], low[i - 2])
                if low[i] < sar:
                    # Flip to bearish
                    bull = False
                    sar  = ep
                    ep   = low[i]
                    af   = initial_af
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + step_af, max_af)
            else:
                sar = max(sar, high[i - 1], high[i - 2])
                if high[i] > sar:
                    # Flip to bullish
                    bull = True
                    sar  = ep
                    ep   = high[i]
                    af   = initial_af
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + step_af, max_af)

            sar_arr[i] = sar

        sar_s  = pd.Series(sar_arr, index=df.index)
        pct    = (pd.Series(close, index=df.index) - sar_s) / sar_s.replace(0.0, np.nan) * 100.0
        signal = (pct / scale).clip(-1.0, 1.0)

        return signal.fillna(0.0).rename("PSAR")
