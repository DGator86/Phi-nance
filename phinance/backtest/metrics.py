"""
phinance.backtest.metrics
==========================

Performance metric calculations for backtest results.

Functions
---------
  bars_per_year(df)            — Infer annualised bar count
  total_return(pv)             — Fractional total return
  cagr(pv, n_bars, bpy)        — Compound annual growth rate
  max_drawdown(pv)             — Maximum peak-to-trough drawdown
  sharpe_ratio(pv, bpy)        — Annualised Sharpe ratio
  sortino_ratio(pv, bpy)       — Annualised Sortino ratio
  win_rate(trades)             — Fraction of profitable trades
  compute_all(pv, df, trades)  — Compute every metric in one call
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

_TRADING_MINUTES_PER_YEAR = 252 * 390  # US equity market


def bars_per_year(df: pd.DataFrame) -> float:
    """Infer the number of bars per calendar year from the DataFrame index.

    For daily bars returns 252.  For intraday bars computes the median
    bar duration so CAGR and Sharpe are correctly annualised regardless
    of the timeframe.

    Parameters
    ----------
    df : pd.DataFrame — OHLCV with DatetimeIndex

    Returns
    -------
    float
    """
    if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
        return 252.0
    deltas = pd.Series(df.index.astype("int64")).diff().dropna()
    if deltas.empty:
        return 252.0
    median_ns = float(deltas.median())
    if median_ns <= 0:
        return 252.0
    median_minutes = median_ns / 60e9
    if median_minutes >= 300:  # ≥ 5 hours → treat as daily
        return 252.0
    return _TRADING_MINUTES_PER_YEAR / median_minutes


def total_return(portfolio_values: np.ndarray, initial_capital: float) -> float:
    """Compute fractional total return.

    Parameters
    ----------
    portfolio_values : array — NAV series
    initial_capital  : float — starting capital

    Returns
    -------
    float — e.g. 0.12 for 12 %
    """
    if initial_capital == 0 or len(portfolio_values) == 0:
        return 0.0
    return float(portfolio_values[-1] - initial_capital) / initial_capital


def cagr(pv: np.ndarray, initial_capital: float, bpy: float) -> float:
    """Compound annual growth rate.

    Parameters
    ----------
    pv              : array — NAV series
    initial_capital : float
    bpy             : float — bars per year

    Returns
    -------
    float
    """
    tr = total_return(pv, initial_capital)
    n_bars = max(len(pv), 1)
    years = n_bars / max(bpy, 1)
    if years <= 0:
        return 0.0
    return float((1 + tr) ** (1 / years) - 1)


def max_drawdown(pv: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (positive fraction).

    Parameters
    ----------
    pv : array — NAV series

    Returns
    -------
    float — e.g. 0.15 for 15 % drawdown
    """
    if len(pv) == 0:
        return 0.0
    peak = np.maximum.accumulate(pv)
    dd = (peak - pv) / (peak + 1e-12)
    return float(np.nanmax(dd))


def sharpe_ratio(pv: np.ndarray, bpy: float = 252.0) -> float:
    """Annualised Sharpe ratio (zero risk-free rate).

    Parameters
    ----------
    pv  : array — NAV series
    bpy : float — bars per year for annualisation

    Returns
    -------
    float
    """
    if len(pv) < 3:
        return 0.0
    rets = np.diff(pv) / (pv[:-1] + 1e-12)
    std = float(np.std(rets))
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(bpy))


def sortino_ratio(pv: np.ndarray, bpy: float = 252.0) -> float:
    """Annualised Sortino ratio (penalises only negative returns).

    Parameters
    ----------
    pv  : array — NAV series
    bpy : float — bars per year

    Returns
    -------
    float
    """
    if len(pv) < 3:
        return 0.0
    rets = np.diff(pv) / (pv[:-1] + 1e-12)
    neg = rets[rets < 0]
    if len(neg) == 0:
        return float(np.mean(rets)) * bpy
    dstd = float(np.std(neg))
    if dstd == 0:
        return 0.0
    return float(np.mean(rets) / dstd * np.sqrt(bpy))


def win_rate(trades: List[Any]) -> float:
    """Fraction of trades with positive P&L.

    Parameters
    ----------
    trades : list — items with a ``.win`` bool attribute or a ``"win"`` key

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not trades:
        return 0.0
    wins = sum(
        (t.win if hasattr(t, "win") else t.get("win", False))
        for t in trades
    )
    return wins / len(trades)


def compute_all(
    portfolio_values: List[float],
    ohlcv: pd.DataFrame,
    initial_capital: float,
    trades: List[Any] | None = None,
) -> Dict[str, float]:
    """Compute every standard performance metric in one call.

    Parameters
    ----------
    portfolio_values : list — NAV series
    ohlcv            : pd.DataFrame — used to determine bar frequency
    initial_capital  : float
    trades           : list, optional — trade objects for win_rate

    Returns
    -------
    dict with keys: total_return, cagr, max_drawdown, sharpe, sortino, win_rate
    """
    pv = np.array(portfolio_values, dtype=float)
    bpy = bars_per_year(ohlcv)

    return {
        "total_return": total_return(pv, initial_capital),
        "cagr":         cagr(pv, initial_capital, bpy),
        "max_drawdown": max_drawdown(pv),
        "sharpe":       sharpe_ratio(pv, bpy),
        "sortino":      sortino_ratio(pv, bpy),
        "win_rate":     win_rate(trades or []),
        "net_pl":       float(pv[-1] - initial_capital) if len(pv) else 0.0,
    }
