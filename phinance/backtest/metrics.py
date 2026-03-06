"""
phinance.backtest.metrics
==========================

Performance metric calculations for backtest results.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _wrap(func: Any) -> Any:
            return func

        return _wrap

_TRADING_MINUTES_PER_YEAR = 252 * 390  # US equity market


@njit(cache=True)
def _max_drawdown_numba(pv: np.ndarray) -> float:
    if pv.size == 0:
        return 0.0
    peak = pv[0]
    max_dd = 0.0
    for i in range(pv.size):
        value = pv[i]
        if value > peak:
            peak = value
        dd = (peak - value) / (peak + 1e-12)
        if dd > max_dd:
            max_dd = dd
    return max_dd


@njit(cache=True)
def _returns_stats_numba(pv: np.ndarray) -> tuple[float, float, float, float, int]:
    n = pv.size
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0, 0

    total = 0.0
    total_sq = 0.0
    neg_total_sq = 0.0
    neg_count = 0

    for i in range(1, n):
        ret = (pv[i] - pv[i - 1]) / (pv[i - 1] + 1e-12)
        total += ret
        total_sq += ret * ret
        if ret < 0:
            neg_total_sq += ret * ret
            neg_count += 1

    count = n - 1
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = np.sqrt(variance)

    if neg_count == 0:
        neg_std = 0.0
    else:
        neg_mean_sq = neg_total_sq / neg_count
        neg_std = np.sqrt(max(neg_mean_sq, 0.0))

    return mean, std, neg_std, total, count


def bars_per_year(df: pd.DataFrame) -> float:
    if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
        return 252.0
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return 252.0
    median_seconds = float(deltas.dt.total_seconds().median())
    if median_seconds <= 0:
        return 252.0
    median_minutes = median_seconds / 60.0
    if median_minutes >= 300:
        return 252.0
    return _TRADING_MINUTES_PER_YEAR / median_minutes


def total_return(portfolio_values: np.ndarray, initial_capital: float) -> float:
    if initial_capital == 0 or len(portfolio_values) == 0:
        return 0.0
    return float(portfolio_values[-1] - initial_capital) / initial_capital


def cagr(pv: np.ndarray, initial_capital: float, bpy: float) -> float:
    tr = total_return(pv, initial_capital)
    n_bars = max(len(pv), 1)
    years = n_bars / max(bpy, 1)
    if years <= 0:
        return 0.0
    return float((1 + tr) ** (1 / years) - 1)


def max_drawdown(pv: np.ndarray) -> float:
    return float(_max_drawdown_numba(np.asarray(pv, dtype=np.float64)))


def sharpe_ratio(pv: np.ndarray, bpy: float = 252.0) -> float:
    mean, std, _, _, count = _returns_stats_numba(np.asarray(pv, dtype=np.float64))
    if count == 0 or std == 0:
        return 0.0
    return float(mean / std * np.sqrt(bpy))


def sortino_ratio(pv: np.ndarray, bpy: float = 252.0) -> float:
    mean, _, neg_std, _, count = _returns_stats_numba(np.asarray(pv, dtype=np.float64))
    if count == 0:
        return 0.0
    if neg_std == 0:
        return float(mean) * bpy
    return float(mean / neg_std * np.sqrt(bpy))


def win_rate(trades: List[Any]) -> float:
    if not trades:
        return 0.0
    wins = sum((t.win if hasattr(t, "win") else t.get("win", False)) for t in trades)
    return wins / len(trades)


def compute_all(
    portfolio_values: List[float],
    ohlcv: pd.DataFrame,
    initial_capital: float,
    trades: List[Any] | None = None,
) -> Dict[str, float]:
    pv = np.asarray(portfolio_values, dtype=np.float64)
    bpy = bars_per_year(ohlcv)

    return {
        "total_return": total_return(pv, initial_capital),
        "cagr": cagr(pv, initial_capital, bpy),
        "max_drawdown": max_drawdown(pv),
        "sharpe": sharpe_ratio(pv, bpy),
        "sortino": sortino_ratio(pv, bpy),
        "win_rate": win_rate(trades or []),
        "net_pl": float(pv[-1] - initial_capital) if len(pv) else 0.0,
    }
