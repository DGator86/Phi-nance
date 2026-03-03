"""
phinance.strategies.cci
========================

CCI (Commodity Channel Index) — volatility/momentum oscillator.

References
----------
* Lambert, D. (1980) original CCI paper
* stockstats: cci  (default period 14)
* arkochhar/Technical-Indicators: CCI()
* Stock.Indicators (.NET): Cci()

Formula
-------
  Typical Price  (TP) = (high + low + close) / 3
  SMA_TP              = SMA(TP, n)
  Mean Absolute Dev   = SMA(|TP − SMA_TP|, n)
  CCI                 = (TP − SMA_TP) / (0.015 × MAD)

  The 0.015 constant ensures ~70–80 % of random price values fall in [−100, +100].

  Signal:
    CCI is normalised by dividing by the `scale` parameter and clipping → [−1, +1]
    Flip sign so that:
      CCI < −threshold → +1 (oversold buy)
      CCI > +threshold → −1 (overbought sell)

Signal convention
-----------------
  +1 → CCI deeply negative (oversold — momentum reversal expected upward)
  −1 → CCI deeply positive (overbought — momentum reversal expected downward)
   0 → CCI near zero (neutral / warmup)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from phinance.strategies.base import BaseIndicator


class CCIIndicator(BaseIndicator):
    """CCI mean-reversion momentum signal.

    Parameters
    ----------
    period : int   — CCI window (default 14)
    scale  : float — normalisation divisor in CCI units (default 100.0)
                     CCI values beyond ±scale map to ±1 signal
    """

    name = "CCI"
    default_params = {"period": 14, "scale": 100.0}
    param_grid = {
        "period": [7, 10, 14, 20],
        "scale":  [75.0, 100.0, 150.0],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int    = 14,
        scale: float   = 100.0,
        **_: Any,
    ) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        tp     = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=period, min_periods=period).mean()
        mad    = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        mad_safe = mad.replace(0.0, np.nan)
        cci    = (tp - sma_tp) / (0.015 * mad_safe)

        # Flip sign: CCI deeply negative → +1 (buy); deeply positive → −1 (sell)
        signal = -(cci / scale).clip(-1.0, 1.0)

        return signal.fillna(0.0).rename("CCI")
