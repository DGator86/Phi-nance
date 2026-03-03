"""
phinance.strategies.ulcer_index
================================

Ulcer Index — downside-risk / drawdown severity indicator.

References
----------
* Peter G. Martin & Byron B. McCann (1989) *The Investor's Guide to
  Fidelity Funds* — original Ulcer Index definition
* Stock.Indicators (.NET) — GetUlcerIndex() series method
* stockstats — ulcer implementation
* https://www.quantifiedstrategies.com/ulcer-index/
* https://vectoralpha.dev/projects/ta/indicators/ui

Formula
-------
  Over a rolling window of *n* bars:

    rolling_max   = max(close[i−n+1 : i+1])
    drawdown_pct  = (close − rolling_max) / rolling_max × 100      [≤ 0]
    squared_dd    = drawdown_pct²
    UI            = sqrt(mean(squared_dd_over_window))

  UI ≥ 0 always; higher values = more volatile downside drawdowns.

  Since a *higher* Ulcer Index means more risk (not a directional signal),
  we invert it to produce a conventional signal:

    signal = −normalize(UI)

  so that a rising Ulcer Index (increasing risk) maps to a negative signal
  and a low UI maps to a near-zero / positive signal.

Signal convention
-----------------
  +1 → UI is very low (calm market, minimal drawdown — favourable)
  −1 → UI is very high (high drawdown risk — unfavourable)
   0 → UI near historical median
  NaN warmup bars are filled with 0 (neutral).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phinance.strategies.base import BaseIndicator


class UlcerIndexIndicator(BaseIndicator):
    """Ulcer Index downside-risk signal.

    Parameters
    ----------
    period : int — rolling window for UI calculation (default 14)

    Notes
    -----
    The signal is the *inverted* normalised Ulcer Index so that higher UI
    (more risk) maps to negative signal values.
    """

    name = "Ulcer Index"
    default_params = {"period": 14}
    param_grid = {
        "period": [7, 10, 14, 20, 28],
    }

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        **_: Any,
    ) -> pd.Series:
        close = df["close"].astype(float)

        n = int(period)
        if n < 1:
            n = 1

        # Rolling maximum over window
        rolling_max = close.rolling(window=n, min_periods=1).max()

        # Percentage drawdown from rolling max (always ≤ 0)
        drawdown_pct = ((close - rolling_max) / rolling_max.where(rolling_max != 0, other=1e-10)) * 100.0

        # Squared drawdown
        squared_dd = drawdown_pct ** 2

        # Ulcer Index = sqrt(mean of squared drawdowns over window)
        ui = squared_dd.rolling(window=n, min_periods=n).mean().apply(np.sqrt)

        # Invert-normalize: high UI → negative signal, low UI → near-zero/positive
        # Use _normalize_abs on (-ui) so sign is preserved
        signal = self._normalize_abs(-ui)

        return signal.fillna(0.0).rename("Ulcer Index")
