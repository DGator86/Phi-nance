"""
phinance.strategies.params
==========================

Default parameter grids for PhiAI optimisation.

Two grids are provided for each indicator:
  DAILY_GRIDS    — Used when timeframe is ``"1D"`` (longer periods, more history)
  INTRADAY_GRIDS — Used for 1m/5m/15m/30m/1H (shorter periods, faster response)

Usage
-----
    from phinance.strategies.params import get_param_grid, DAILY_GRIDS, INTRADAY_GRIDS

    grid = get_param_grid("RSI", timeframe="1D")
    # {'period': [7, 14, 21], 'oversold': [25, 30, 35], 'overbought': [65, 70, 75]}

    grid = get_param_grid("MACD", timeframe="15m")
    # {'fast_period': [5, 8, 12], 'slow_period': [13, 21, 26], 'signal_period': [5, 7, 9]}

Sources
-------
  Defaults drawn from:
    * Wilder (1978) — RSI, ATR, PSAR
    * Appel (1979) — MACD
    * Bollinger (2002) — Bollinger Bands
    * Lane (1957) — Stochastic
    * Williams (1973) — Williams %R
    * Lambert (1980) — CCI
    * stockstats default windows — OBV, VWAP
    * Stock.Indicators (.NET) default parameters
"""

from __future__ import annotations

from typing import Any, Dict

_INTRADAY_TF = {"1m", "5m", "15m", "30m", "1H"}

# ── Daily parameter grids  ────────────────────────────────────────────────────

DAILY_GRIDS: Dict[str, Dict[str, list]] = {

    "RSI": {
        "period":     [7, 9, 14, 21, 28],
        "oversold":   [25, 30, 35],
        "overbought": [65, 70, 75],
    },

    "MACD": {
        "fast_period":   [8, 10, 12, 16],
        "slow_period":   [21, 26, 30, 34],
        "signal_period": [7, 9, 11],
    },

    "Bollinger": {
        "period":  [10, 15, 20, 25, 30],
        "num_std": [1.5, 2.0, 2.5],
    },

    "Dual SMA": {
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100, 200],
    },

    "EMA Cross": {
        "fast_period": [5, 8, 12, 20],
        "slow_period": [20, 26, 34, 50, 100],
    },

    "Mean Reversion": {
        "period":      [10, 20, 30, 50],
        "z_threshold": [1.5, 2.0, 2.5],
    },

    "Breakout": {
        "period": [10, 15, 20, 30, 50],
    },

    "Buy & Hold": {},

    "VWAP": {
        "period":   [10, 20, 30, 50],
        "band_pct": [0.25, 0.5, 1.0, 1.5],
    },

    "ATR": {
        "period":      [7, 10, 14, 20],
        "lookback":    [30, 50, 100],
        "z_threshold": [1.5, 2.0, 2.5],
    },

    "Stochastic": {
        "k_period":   [9, 14, 21],
        "d_period":   [3, 5],
        "smooth":     [1, 3],
        "oversold":   [15.0, 20.0, 25.0],
        "overbought": [75.0, 80.0, 85.0],
    },

    "Williams %R": {
        "period":     [7, 10, 14, 21],
        "oversold":   [-85.0, -80.0, -75.0],
        "overbought": [-25.0, -20.0, -15.0],
    },

    "CCI": {
        "period": [7, 10, 14, 20],
        "scale":  [75.0, 100.0, 150.0],
    },

    "OBV": {
        "period": [7, 10, 14, 21, 30],
    },

    "PSAR": {
        "initial_af": [0.01, 0.02, 0.03],
        "step_af":    [0.01, 0.02, 0.03],
        "max_af":     [0.10, 0.20, 0.30],
    },
}

# ── Intraday parameter grids  ─────────────────────────────────────────────────
# Shorter windows for faster signal response on sub-daily timeframes.

INTRADAY_GRIDS: Dict[str, Dict[str, list]] = {

    "RSI": {
        "period":     [5, 7, 9, 14],
        "oversold":   [25, 30, 35],
        "overbought": [65, 70, 75],
    },

    "MACD": {
        "fast_period":   [5, 8, 10, 12],
        "slow_period":   [13, 18, 21, 26],
        "signal_period": [5, 7, 9],
    },

    "Bollinger": {
        "period":  [7, 10, 14, 20],
        "num_std": [1.5, 2.0, 2.5],
    },

    "Dual SMA": {
        "fast_period": [3, 5, 8, 10],
        "slow_period": [15, 20, 30, 50],
    },

    "EMA Cross": {
        "fast_period": [3, 5, 8, 12],
        "slow_period": [13, 20, 26, 34],
    },

    "Mean Reversion": {
        "period":      [5, 10, 14, 20],
        "z_threshold": [1.5, 2.0, 2.5],
    },

    "Breakout": {
        "period": [5, 8, 10, 14, 20],
    },

    "Buy & Hold": {},

    "VWAP": {
        "period":   [5, 10, 20],
        "band_pct": [0.1, 0.2, 0.3, 0.5],
    },

    "ATR": {
        "period":      [5, 7, 10, 14],
        "lookback":    [20, 30, 50],
        "z_threshold": [1.5, 2.0, 2.5],
    },

    "Stochastic": {
        "k_period":   [5, 9, 14],
        "d_period":   [3, 5],
        "smooth":     [1, 3],
        "oversold":   [15.0, 20.0, 25.0],
        "overbought": [75.0, 80.0, 85.0],
    },

    "Williams %R": {
        "period":     [5, 7, 10, 14],
        "oversold":   [-85.0, -80.0, -75.0],
        "overbought": [-25.0, -20.0, -15.0],
    },

    "CCI": {
        "period": [5, 7, 10, 14],
        "scale":  [75.0, 100.0, 150.0],
    },

    "OBV": {
        "period": [5, 7, 10, 14],
    },

    "PSAR": {
        "initial_af": [0.01, 0.02, 0.03],
        "step_af":    [0.01, 0.02, 0.03],
        "max_af":     [0.10, 0.20, 0.30],
    },
}


# ── Public helper  ────────────────────────────────────────────────────────────


def get_param_grid(name: str, timeframe: str = "1D") -> Dict[str, list]:
    """Return the optimisation parameter grid for a named indicator.

    Parameters
    ----------
    name      : str — indicator name (key in INDICATOR_CATALOG)
    timeframe : str — ``"1D"`` for daily grid; any intraday TF for intraday grid

    Returns
    -------
    dict — ``{param_name: [list_of_values]}``; empty dict if name not found
    """
    grids = INTRADAY_GRIDS if timeframe in _INTRADAY_TF else DAILY_GRIDS
    return dict(grids.get(name, {}))


def default_params_for(name: str, timeframe: str = "1D") -> Dict[str, Any]:
    """Return the mid-point (first reasonable) default params for an indicator.

    These are the *starting* parameters before PhiAI optimisation.

    Parameters
    ----------
    name      : str
    timeframe : str

    Returns
    -------
    dict — one value per parameter (first item in each grid list)
    """
    grid = get_param_grid(name, timeframe)
    return {k: v[len(v) // 2] for k, v in grid.items()}  # pick middle of grid
