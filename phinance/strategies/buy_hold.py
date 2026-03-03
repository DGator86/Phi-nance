"""
phinance.strategies.buy_hold
=============================

Buy-and-Hold baseline indicator — always returns a constant weak buy signal.

References
----------
* phi/indicators/simple.py original implementation

Formula
-------
  signal = +0.5 (constant)  — always slightly long, acts as a benchmark

Signal convention
-----------------
  +0.5 → constant mild buy signal (benchmark / baseline comparison)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinance.strategies.base import BaseIndicator


class BuyHoldIndicator(BaseIndicator):
    """Constant buy-and-hold baseline signal (+0.5 every bar).

    Used as a benchmark to compare against active indicator strategies.
    """

    name = "Buy & Hold"
    default_params: dict = {}
    param_grid: dict = {}

    def compute(self, df: pd.DataFrame, **_: Any) -> pd.Series:
        return pd.Series(0.5, index=df.index, dtype=float).rename("Buy & Hold")
