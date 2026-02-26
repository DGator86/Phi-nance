"""
Direct vectorized backtest — no Lumibot, no datasource.
Uses OHLCV DataFrame directly. Guaranteed to work with pipeline data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def run_direct_backtest(
    ohlcv: pd.DataFrame,
    symbol: str,
    indicators: Dict[str, Dict[str, Any]],
    blend_weights: Dict[str, float],
    blend_method: str = "weighted_sum",
    signal_threshold: float = 0.15,
    initial_capital: float = 100_000,
) -> tuple[Dict[str, Any], Any]:
    """
    Run backtest directly on OHLCV. Returns (results_dict, strat_like_object).

    results_dict: total_return, cagr, max_drawdown, sharpe, portfolio_value
    strat_like: object with .prediction_log for accuracy display
    """
    df = ohlcv.copy()
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    for r in required:
        if r not in cols:
            raise ValueError(f"OHLCV missing column: {r}")
    df = df.rename(columns={cols[r]: r for r in required})[required]

    # Compute indicators
    from phi.indicators.simple import compute_indicator, INDICATOR_COMPUTERS

    signals_dict = {}
    for name, cfg in indicators.items():
        if name not in INDICATOR_COMPUTERS:
            continue
        params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
        try:
            sig = compute_indicator(name, df, params)
            if sig is not None and not sig.empty:
                signals_dict[name] = sig
        except Exception:
            pass

    if not signals_dict:
        return _empty_results(initial_capital), _empty_strat()

    signals_df = pd.DataFrame(signals_dict)
    signals_df = signals_df.reindex(df.index).ffill().bfill()

    from phi.blending import blend_signals

    composite = blend_signals(
        signals_df,
        weights=blend_weights,
        method=blend_method,
        regime_probs=None,
    )
    if composite.empty:
        return _empty_results(initial_capital), _empty_strat()

    # Simulate bar-by-bar
    cap = float(initial_capital)
    position = 0  # shares
    entry_price = 0.0
    portfolio_values: List[float] = [cap]
    prediction_log: List[Dict] = []
    closes = df["close"].values

    for i in range(len(composite)):
        sig = composite.iloc[i]
        price = float(closes[i])
        if np.isnan(price) or price <= 0:
            portfolio_values.append(cap)
            continue

        if sig > signal_threshold:
            direction = "UP"
            if position == 0:
                qty = int(cap * 0.95 // price)
                if qty > 0:
                    position = qty
                    entry_price = price
                    cap -= qty * price
        elif sig < -signal_threshold:
            direction = "DOWN"
            if position > 0:
                cap += position * price
                position = 0
        else:
            direction = "NEUTRAL"

        pv = cap + position * price
        portfolio_values.append(pv)
        prediction_log.append({
            "date": df.index[i],
            "symbol": symbol,
            "signal": direction,
            "price": price,
        })

    # Close any remaining position at last price
    if position > 0:
        cap += position * float(closes[-1])
        position = 0

    pv_series = np.array(portfolio_values)
    returns = np.diff(pv_series) / (pv_series[:-1] + 1e-12)
    total_return = (pv_series[-1] - initial_capital) / initial_capital if initial_capital else 0
    years = len(df) / 252.0 if len(df) > 0 else 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(pv_series)
    dd = (peak - pv_series) / (peak + 1e-12)
    max_drawdown = float(np.nanmax(dd)) if len(dd) > 0 else 0.0

    # Sharpe (annualized)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    results = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "portfolio_value": list(pv_series),
        "net_pl": pv_series[-1] - initial_capital,
    }

    # Strat-like object for _display_results / compute_prediction_accuracy
    strat = type("Strat", (), {"prediction_log": prediction_log, "_prediction_log": prediction_log})()

    return results, strat


def _empty_results(cap: float) -> Dict[str, Any]:
    return {
        "total_return": 0,
        "cagr": 0,
        "max_drawdown": 0,
        "sharpe": 0,
        "portfolio_value": [cap],
        "net_pl": 0,
    }


def _empty_strat() -> Any:
    return type("Strat", (), {"prediction_log": []})()
