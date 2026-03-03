"""
phinance.backtest.engine
=========================

Core vectorised backtest simulation loop.

Operates directly on normalised OHLCV + composite signal — no external
framework dependencies. Simulates bar-by-bar execution:

  BUY  when signal > signal_threshold  (and flat)
  SELL when signal < -signal_threshold (and long)

Returns a raw (portfolio_values, prediction_log, trades) tuple.

This module is intentionally low-level; most callers should use
``phinance.backtest.runner.run_backtest()`` instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from phinance.backtest.models import Trade
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


def simulate(
    ohlcv: pd.DataFrame,
    composite_signal: pd.Series,
    symbol: str = "",
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.95,
    signal_threshold: float = 0.15,
) -> Tuple[List[float], List[Dict[str, Any]], List[Trade]]:
    """Execute a vectorised bar-by-bar backtest simulation.

    Parameters
    ----------
    ohlcv             : pd.DataFrame — OHLCV data
    composite_signal  : pd.Series   — blended signal aligned to ohlcv.index
    symbol            : str         — ticker name for logging / trade records
    initial_capital   : float       — starting NAV
    position_size_pct : float       — fraction of capital to deploy per trade
    signal_threshold  : float       — minimum signal magnitude to act on

    Returns
    -------
    (portfolio_values, prediction_log, trades)

    portfolio_values : List[float]        — NAV at each bar
    prediction_log   : List[Dict]         — bar-by-bar signal records
    trades           : List[Trade]        — closed round-trip trades
    """
    closes = ohlcv["close"].values
    cap = float(initial_capital)
    position = 0       # shares
    entry_price = 0.0
    entry_date: Any = None
    entry_bar = 0

    portfolio_values: List[float] = [cap]
    prediction_log:   List[Dict[str, Any]] = []
    trades:           List[Trade] = []

    for i, sig in enumerate(composite_signal):
        price = float(closes[i])
        date = ohlcv.index[i]

        if np.isnan(price) or price <= 0:
            portfolio_values.append(cap + position * (float(closes[i - 1]) if i > 0 else 0))
            continue

        if sig > signal_threshold and position == 0:
            # Enter long
            qty = int(cap * position_size_pct // price)
            if qty > 0:
                position    = qty
                entry_price = price
                entry_date  = date
                entry_bar   = i
                cap        -= qty * price
            direction = "UP"

        elif sig < -signal_threshold and position > 0:
            # Exit long
            cap += position * price
            pnl_abs = position * (price - entry_price)
            pnl_pct = (price - entry_price) / entry_price if entry_price else 0.0
            hold    = i - entry_bar
            trades.append(Trade(
                entry_date  = entry_date,
                exit_date   = date,
                symbol      = symbol,
                entry_price = entry_price,
                exit_price  = price,
                quantity    = position,
                pnl         = pnl_abs,
                pnl_pct     = pnl_pct,
                hold_bars   = hold,
            ))
            position = 0
            direction = "DOWN"

        else:
            direction = "NEUTRAL"

        pv = cap + position * price
        portfolio_values.append(pv)
        prediction_log.append({
            "date":   date,
            "symbol": symbol,
            "signal": direction,
            "price":  price,
        })

    # Close any remaining position at last bar
    if position > 0:
        last_price = float(closes[-1])
        cap += position * last_price
        if entry_price > 0:
            trades.append(Trade(
                entry_date  = entry_date,
                exit_date   = ohlcv.index[-1],
                symbol      = symbol,
                entry_price = entry_price,
                exit_price  = last_price,
                quantity    = position,
                pnl         = position * (last_price - entry_price),
                pnl_pct     = (last_price - entry_price) / entry_price,
                hold_bars   = len(composite_signal) - 1 - entry_bar,
            ))
        position = 0

    # Replace the first placeholder with final capital
    portfolio_values[0] = float(initial_capital)
    return portfolio_values, prediction_log, trades
