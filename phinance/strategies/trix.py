"""
phinance.strategies.trix
=========================

TRIX — Triple Smoothed EMA Momentum Oscillator.

References
----------
* Hutson, J. K. (1983) — original TRIX article, *Stocks & Commodities*
* Stock.Indicators (.NET) — GetTrix() series method
* python.stockindicators.dev/indicators/Trix/
* arkochhar/Technical-Indicators — TRIX implementation
* https://www.luxalgo.com/blog/trix-triple-exponential-moving-average-guide/

Formula
-------
  EMA1 = EMA(close, n)
  EMA2 = EMA(EMA1,  n)
  EMA3 = EMA(EMA2,  n)

  TRIX = (EMA3 − EMA3.shift(1)) / EMA3.shift(1) × 100   [% change]

  Signal line = EMA(TRIX, signal_period)

  TRIX oscillates around zero:
    positive TRIX → upward momentum
    negative TRIX → downward momentum
    crossover with signal line → entry/exit

  The triple smoothing acts as a low-pass filter that effectively
  eliminates noise and short-cycle fluctuations.

Signal convention
-----------------
  +1 → TRIX strongly positive (upward triple-smoothed momentum)
  −1 → TRIX strongly negative (downward triple-smoothed momentum)
   0 → TRIX near zero (flat/consolidating)
  NaN warmup bars are filled with 0 (neutral).

  The raw TRIX value (not the TRIX − signal histogram) is used so that
  the signal directly reflects the direction of the triple-smoothed
  momentum rather than its acceleration/deceleration cross.
  The optional signal_period is retained for the ``signal`` attribute
  (available for crossover detection) but is not used in the primary
  signal generation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class TRIXIndicator(BaseIndicator):
    """TRIX triple-smoothed EMA momentum oscillator.

    Parameters
    ----------
    period : int — EMA smoothing period (default 15)
    signal : int — signal-line EMA period (default 9)

    Notes
    -----
    Classic TRIX uses period = 15 and signal = 9 (daily).
    Shorter periods (8–12) suit faster, noisier markets.
    """

    name = "TRIX"
    default_params = {"period": 15, "signal": 9}
    param_grid = {
        "period": [8, 10, 12, 15, 18, 21],
        "signal": [7,  9, 12],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 15,
        signal: int = 9,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)

        n = int(period)
        s = int(signal)
        if n < 1:
            n = 1
        if s < 1:
            s = 1

        # Triple-smoothed EMA
        ema1 = close.ewm(span=n, min_periods=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n,  min_periods=n, adjust=False).mean()
        ema3 = ema2.ewm(span=n,  min_periods=n, adjust=False).mean()

        # Percentage rate-of-change of EMA3
        ema3_prev = ema3.shift(1)
        trix = (
            (ema3 - ema3_prev) / ema3_prev.where(ema3_prev != 0, other=1e-10)
        ) * 100.0

        # Use raw TRIX as the primary signal (not TRIX − signal_line histogram).
        # Raw TRIX > 0 in uptrend, < 0 in downtrend — gives stable directional bias.
        # The signal_line parameter is kept for optional crossover use externally.
        result = self._normalize_abs(trix)

        return result.fillna(0.0).rename("TRIX")
