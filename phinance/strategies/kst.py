"""
phinance.strategies.kst
========================

KST (Know Sure Thing) — Martin Pring's multi-period momentum oscillator.

References
----------
* Pring, M. J. (1992) — original KST concept, *Stocks & Commodities* magazine
* Stock.Indicators (.NET) — GetKst() series method
* stockstats — KST implementation
* https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/prings-know-sure-thing-kst
* https://www.investopedia.com/terms/k/know-sure-thing-kst.asp

Formula
-------
  KST combines four Rate-of-Change (ROC) components, each smoothed by an SMA,
  and weights them to produce a single momentum oscillator.

  ROC(n)     = (close / close.shift(n) − 1) × 100

  Standard daily parameters (Martin Pring's originals):
    rcma1 = SMA(ROC(10), 10)   weight = 1
    rcma2 = SMA(ROC(15), 10)   weight = 2
    rcma3 = SMA(ROC(20), 10)   weight = 3
    rcma4 = SMA(ROC(30), 15)   weight = 4

  KST = rcma1×1 + rcma2×2 + rcma3×3 + rcma4×4

  Signal line (optional) = SMA(KST, signal_period)

  For the Phi-nance signal we use:
    positive KST above its signal line → bullish  (+direction)
    negative KST below its signal line → bearish  (−direction)

Signal convention
-----------------
  +1 → KST strongly positive (bullish multi-period momentum)
  −1 → KST strongly negative (bearish multi-period momentum)
   0 → KST near zero (neutral momentum)
  NaN warmup bars are filled with 0 (neutral).

  The raw KST oscillator value is used as the signal (not the KST − signal
  histogram), providing a stable directional momentum reading.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class KSTIndicator(BaseIndicator):
    """KST momentum oscillator using four smoothed ROC components.

    Parameters
    ----------
    roc1       : int — ROC period for component 1 (default 10)
    roc2       : int — ROC period for component 2 (default 15)
    roc3       : int — ROC period for component 3 (default 20)
    roc4       : int — ROC period for component 4 (default 30)
    sma1       : int — SMA smoothing of ROC 1 (default 10)
    sma2       : int — SMA smoothing of ROC 2 (default 10)
    sma3       : int — SMA smoothing of ROC 3 (default 10)
    sma4       : int — SMA smoothing of ROC 4 (default 15)
    signal     : int — signal line SMA period (default 9)
    """

    name = "KST"
    default_params = {
        "roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30,
        "sma1": 10, "sma2": 10, "sma3": 10, "sma4": 15,
        "signal": 9,
    }
    param_grid = {
        "roc1": [8, 10, 12],
        "roc4": [25, 30, 35],
        "signal": [7, 9, 12],
    }

    def compute(
        self,
        df: pd.DataFrame,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15,
        signal: int = 9,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)

        def _roc(n: int) -> pd.Series:
            """Rate of change as percentage."""
            shifted = close.shift(n)
            return (close / shifted.where(shifted != 0, other=1e-10) - 1.0) * 100.0

        def _sma(s: pd.Series, n: int) -> pd.Series:
            return s.rolling(window=n, min_periods=n).mean()

        # Four smoothed ROC components
        rcma1 = _sma(_roc(int(roc1)), int(sma1))
        rcma2 = _sma(_roc(int(roc2)), int(sma2))
        rcma3 = _sma(_roc(int(roc3)), int(sma3))
        rcma4 = _sma(_roc(int(roc4)), int(sma4))

        # Weighted sum — Pring's standard weights 1, 2, 3, 4
        kst = rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4

        # Use raw KST as the primary signal.
        # KST > 0 in uptrend, < 0 in downtrend — stable directional bias.
        # The signal_line parameter is retained for optional crossover detection.
        result = self._normalize_abs(kst)

        return result.fillna(0.0).rename("KST")
