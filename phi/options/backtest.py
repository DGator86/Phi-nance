"""
Options Backtest — Simplified Delta-Based Simulation

Simulates long call/put P&L using underlying OHLCV and delta approximation.
Uses MarketDataApp options chain snapshot when available for delta anchoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_data import get_marketdataapp_snapshot


def compute_greeks(
    delta: float,
    gamma: float = None,
    theta: float = None,
    vega: float = None,
) -> dict:
    """Return a dictionary of option Greeks.

    Parameters
    ----------
    delta : float
        Rate of change of option price with respect to underlying price.
        Positive for calls (0 to 1), negative for puts (-1 to 0).
    gamma : float, optional
        Rate of change of delta with respect to underlying price.
    theta : float, optional
        Time decay — daily dollar decay of the option value.
    vega : float, optional
        Sensitivity to 1 percentage-point change in implied volatility.

    Returns
    -------
    dict
        ``{"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}``

    Examples
    --------
    >>> compute_greeks(0.5, gamma=0.05, theta=-0.02, vega=0.10)
    {'delta': 0.5, 'gamma': 0.05, 'theta': -0.02, 'vega': 0.10}
    """
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }


def run_options_backtest(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    strategy_type: str = "long_call",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.1,
    delta_assumption: float = 0.5,
    exit_profit_pct: float = 0.5,
    exit_stop_pct: float = -0.3,
) -> dict:
    """
    Simplified options backtest using delta approximation.

    P&L per bar = notional * delta * (price_return)
    where notional = capital * position_pct at entry.

    Parameters
    ----------
    ohlcv : DataFrame with close, index = datetime
    strategy_type : "long_call" | "long_put"
    initial_capital : starting capital
    position_pct : fraction of capital in option notional (0.1 = 10%)
    delta_assumption : assumed delta (0.5 = ATM)
    exit_profit_pct : exit at +50% on position
    exit_stop_pct : exit at -30% on position

    Returns
    -------
    dict with keys: portfolio_value, trades, total_return, cagr, max_drawdown, sharpe
    """
    close = ohlcv["close"].values
    if len(close) < 2:
        return {"portfolio_value": [initial_capital], "total_return": 0, "cagr": 0, "max_drawdown": 0, "sharpe": 0}

    mult = 1.0 if strategy_type == "long_call" else -1.0

    # Try to anchor delta with an actual options chain snapshot from MarketDataApp.
    spot = float(close[0])
    snap = get_marketdataapp_snapshot(symbol=symbol, spot_price=spot, option_type="call" if mult > 0 else "put")
    if snap and snap.delta is not None:
        delta_assumption = float(abs(snap.delta))

    returns = np.diff(close) / close[:-1] * mult

    capital = initial_capital
    notional = capital * position_pct
    position_pnl = 0.0
    in_position = True
    pv_series = [capital]

    for i, r in enumerate(returns):
        if not in_position:
            pv_series.append(capital)
            continue

        delta_pnl = notional * delta_assumption * r
        position_pnl += delta_pnl
        capital += delta_pnl

        cum_ret = position_pnl / (notional * delta_assumption) if notional else 0
        if cum_ret >= exit_profit_pct or cum_ret <= exit_stop_pct:
            in_position = False

        pv_series.append(capital)

    pv_series = np.array(pv_series)
    total_return = (pv_series[-1] / initial_capital - 1) if initial_capital else 0

    # CAGR
    n_years = len(ohlcv) / 252 if len(ohlcv) > 252 else 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else total_return

    # Max drawdown
    peak = np.maximum.accumulate(pv_series)
    dd = (pv_series - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd)) if len(dd) else 0

    # Sharpe (annualized)
    pv_ret = np.diff(pv_series) / np.maximum(pv_series[:-1], 1e-8)
    sharpe = float(np.mean(pv_ret) / np.std(pv_ret) * np.sqrt(252)) if np.std(pv_ret) > 0 else 0

    out = {
        "portfolio_value": list(pv_series),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": [],
    }
    if snap:
        out["options_snapshot"] = {
            "source": snap.source,
            "strike": snap.strike,
            "expiry": snap.expiry,
            "mid": snap.mid,
            "delta": snap.delta,
            "implied_volatility": snap.implied_volatility,
        }
    return out
