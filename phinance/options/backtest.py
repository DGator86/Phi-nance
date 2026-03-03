"""
phinance.options.backtest
==========================

Simplified delta-based options backtest simulation.

Simulates long call/put P&L using underlying OHLCV and a delta approximation:
  P&L per bar = notional * delta * (close_return)

Optionally anchors the delta assumption to a live MarketDataApp snapshot.

Usage
-----
    from phinance.options.backtest import run_options_backtest

    result = run_options_backtest(
        ohlcv           = df,
        symbol          = "SPY",
        strategy_type   = "long_call",
        initial_capital = 100_000,
    )
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from phinance.options.market_data import get_marketdataapp_snapshot


def run_options_backtest(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    strategy_type: str = "long_call",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.10,
    delta_assumption: float = 0.50,
    exit_profit_pct: float = 0.50,
    exit_stop_pct: float = -0.30,
) -> Dict[str, Any]:
    """Run a simplified delta-based options backtest.

    Parameters
    ----------
    ohlcv             : pd.DataFrame — OHLCV with close column
    symbol            : str          — ticker (used for market data lookup)
    strategy_type     : str          — ``"long_call"`` | ``"long_put"``
    initial_capital   : float
    position_pct      : float        — fraction of capital in option notional
    delta_assumption  : float        — default delta (0.5 = ATM)
    exit_profit_pct   : float        — exit at this gain fraction
    exit_stop_pct     : float        — exit at this loss fraction (negative)

    Returns
    -------
    dict with keys: portfolio_value, total_return, cagr, max_drawdown,
                    sharpe, trades, [options_snapshot]
    """
    close = ohlcv["close"].values
    if len(close) < 2:
        return {
            "portfolio_value": [initial_capital],
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trades": [],
        }

    mult = 1.0 if strategy_type == "long_call" else -1.0

    # Anchor delta with a live snapshot if available
    spot = float(close[0])
    snap = get_marketdataapp_snapshot(
        symbol=symbol,
        spot_price=spot,
        option_type="call" if mult > 0 else "put",
    )
    if snap and snap.delta is not None:
        delta_assumption = float(abs(snap.delta))

    returns = np.diff(close) / close[:-1] * mult

    capital = initial_capital
    notional = capital * position_pct
    position_pnl = 0.0
    in_position = True
    pv_series = [capital]

    for r in returns:
        if not in_position:
            pv_series.append(capital)
            continue

        delta_pnl = notional * delta_assumption * r
        position_pnl += delta_pnl
        capital += delta_pnl

        cum_ret = (
            position_pnl / (notional * delta_assumption)
            if notional and delta_assumption
            else 0.0
        )
        if cum_ret >= exit_profit_pct or cum_ret <= exit_stop_pct:
            in_position = False

        pv_series.append(capital)

    pv = np.array(pv_series)
    tr = (pv[-1] / initial_capital - 1) if initial_capital else 0.0

    n_years = len(ohlcv) / 252 if len(ohlcv) > 252 else 1
    _cagr = (1 + tr) ** (1 / n_years) - 1 if n_years > 0 else tr

    peak = np.maximum.accumulate(pv)
    dd_arr = (pv - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd_arr)) if len(dd_arr) else 0.0

    pv_ret = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    std = float(np.std(pv_ret))
    sharpe = float(np.mean(pv_ret) / std * np.sqrt(252)) if std > 0 else 0.0

    out: Dict[str, Any] = {
        "portfolio_value": list(pv),
        "total_return":    float(tr),
        "cagr":            float(_cagr),
        "max_drawdown":    float(max_dd),
        "sharpe":          float(sharpe),
        "trades":          [],
    }
    if snap:
        out["options_snapshot"] = {
            "source":             snap.source,
            "strike":             snap.strike,
            "expiry":             snap.expiry,
            "mid":                snap.mid,
            "delta":              snap.delta,
            "implied_volatility": snap.implied_volatility,
        }
    return out
