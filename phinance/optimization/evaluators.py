"""
phinance.optimization.evaluators
==================================

Fast objective-function implementations for PhiAI optimisation.

Each function returns a scalar ``score`` where **higher is better**.

Functions
---------
  direction_accuracy(ohlcv, indicator_name, params)
      — fraction of bars where sign(signal) == sign(next_close_return)
  sharpe_proxy(portfolio_values)
      — annualised Sharpe-like score from a portfolio value series
  sortino_proxy(portfolio_values)
      — Sortino-like score (downside deviation only)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def direction_accuracy(
    ohlcv: pd.DataFrame,
    indicator_name: str,
    params: Dict[str, Any],
) -> float:
    """Compute the directional-accuracy of an indicator's signal.

    Measures what fraction of signal predictions correctly identify
    the next bar's price direction (up/down).

    Parameters
    ----------
    ohlcv          : pd.DataFrame — OHLCV data
    indicator_name : str — key in INDICATOR_CATALOG
    params         : dict — indicator parameters

    Returns
    -------
    float — accuracy in [0.0, 1.0] (0.5 = random)
    """
    from phinance.strategies.indicator_catalog import compute_indicator

    try:
        sig = compute_indicator(indicator_name, ohlcv, params)
    except Exception:
        return 0.0

    if sig is None or len(sig) < 10:
        return 0.0

    close = ohlcv["close"].values
    direction = np.zeros(len(close) - 1)
    direction[close[1:] > close[:-1]] = 1
    direction[close[1:] < close[:-1]] = -1

    sig_values = sig.iloc[:-1].values
    n = min(len(sig_values), len(direction))
    sig_trimmed = sig_values[-n:]
    dir_trimmed = direction[-n:]

    predicted = np.sign(sig_trimmed)
    predicted[predicted == 0] = 1  # Treat flat as bullish

    matches = np.sum((predicted * dir_trimmed) > 0)
    return float(matches / max(n, 1))


def sharpe_proxy(portfolio_values: np.ndarray) -> float:
    """Annualised Sharpe-ratio proxy from a portfolio value series.

    Uses 252 bars/year and zero risk-free rate.

    Parameters
    ----------
    portfolio_values : array-like — portfolio NAV over time

    Returns
    -------
    float — Sharpe ratio (higher is better)
    """
    pv = np.asarray(portfolio_values, dtype=float)
    if len(pv) < 3:
        return 0.0
    rets = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    std = float(np.std(rets))
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252))


def sortino_proxy(portfolio_values: np.ndarray) -> float:
    """Sortino-ratio proxy (penalises only negative returns).

    Parameters
    ----------
    portfolio_values : array-like

    Returns
    -------
    float
    """
    pv = np.asarray(portfolio_values, dtype=float)
    if len(pv) < 3:
        return 0.0
    rets = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    neg_rets = rets[rets < 0]
    if len(neg_rets) == 0:
        return float(np.mean(rets)) * 252
    downside_std = float(np.std(neg_rets))
    if downside_std == 0:
        return 0.0
    return float(np.mean(rets) / downside_std * np.sqrt(252))
