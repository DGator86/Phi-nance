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

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _wrap(func: Any) -> Any:
            return func

        return _wrap

from phinance.backtest.models import Trade
from phinance.data.streaming_loader import StreamingDataLoader
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


@njit(cache=True)
def _simulate_state(
    closes: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    position_size_pct: float,
    signal_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated state transition core for backtest simulation."""
    n = len(closes)
    portfolio_values = np.empty(n + 1, dtype=np.float64)
    portfolio_values[0] = initial_capital

    entry_idx = np.full(n, -1, dtype=np.int64)
    exit_idx = np.full(n, -1, dtype=np.int64)
    qty_arr = np.zeros(n, dtype=np.int64)
    entry_price_arr = np.zeros(n, dtype=np.float64)
    exit_price_arr = np.zeros(n, dtype=np.float64)
    hold_bars_arr = np.zeros(n, dtype=np.int64)

    cap = initial_capital
    position = 0
    open_entry_idx = -1
    open_entry_price = 0.0
    trade_count = 0

    for i in range(n):
        price = closes[i]
        sig = signals[i]

        if np.isnan(price) or price <= 0:
            prev_price = closes[i - 1] if i > 0 else 0.0
            portfolio_values[i + 1] = cap + position * prev_price
            continue

        if sig > signal_threshold and position == 0:
            qty = int((cap * position_size_pct) // price)
            if qty > 0:
                position = qty
                open_entry_price = price
                open_entry_idx = i
                cap -= qty * price

        elif sig < -signal_threshold and position > 0:
            cap += position * price
            if open_entry_idx >= 0 and trade_count < n:
                entry_idx[trade_count] = open_entry_idx
                exit_idx[trade_count] = i
                qty_arr[trade_count] = position
                entry_price_arr[trade_count] = open_entry_price
                exit_price_arr[trade_count] = price
                hold_bars_arr[trade_count] = i - open_entry_idx
                trade_count += 1
            position = 0
            open_entry_idx = -1
            open_entry_price = 0.0

        portfolio_values[i + 1] = cap + position * price

    if position > 0:
        last_price = closes[n - 1]
        cap += position * last_price
        if open_entry_idx >= 0 and trade_count < n and open_entry_price > 0:
            entry_idx[trade_count] = open_entry_idx
            exit_idx[trade_count] = n - 1
            qty_arr[trade_count] = position
            entry_price_arr[trade_count] = open_entry_price
            exit_price_arr[trade_count] = last_price
            hold_bars_arr[trade_count] = n - 1 - open_entry_idx
            trade_count += 1
        portfolio_values[n] = cap

    return (
        portfolio_values,
        entry_idx[:trade_count],
        exit_idx[:trade_count],
        qty_arr[:trade_count],
        entry_price_arr[:trade_count],
        exit_price_arr[:trade_count],
        hold_bars_arr[:trade_count],
    )


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
    closes = ohlcv["close"].to_numpy(dtype=np.float64, copy=False)
    signals = composite_signal.reindex(ohlcv.index).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    dates = ohlcv.index.to_list()

    (
        portfolio_values,
        entry_idx,
        exit_idx,
        qty_arr,
        entry_price_arr,
        exit_price_arr,
        hold_bars_arr,
    ) = _simulate_state(
        closes,
        signals,
        float(initial_capital),
        float(position_size_pct),
        float(signal_threshold),
    )

    prediction_log: List[Dict[str, Any]] = []
    for i, sig in enumerate(signals):
        direction = "UP" if sig > signal_threshold else "DOWN" if sig < -signal_threshold else "NEUTRAL"
        prediction_log.append(
            {
                "date": dates[i],
                "symbol": symbol,
                "signal": direction,
                "price": float(closes[i]),
            }
        )

    trades: List[Trade] = []
    for i in range(len(entry_idx)):
        ent_i = int(entry_idx[i])
        ex_i = int(exit_idx[i])
        qty = int(qty_arr[i])
        entry_price = float(entry_price_arr[i])
        exit_price = float(exit_price_arr[i])
        pnl_abs = qty * (exit_price - entry_price)
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        trades.append(
            Trade(
                entry_date=dates[ent_i],
                exit_date=dates[ex_i],
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl_abs,
                pnl_pct=pnl_pct,
                hold_bars=int(hold_bars_arr[i]),
            )
        )

    return portfolio_values.tolist(), prediction_log, trades


def simulate_streaming(
    ohlcv: pd.DataFrame,
    composite_signal: pd.Series,
    symbol: str = "",
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.95,
    signal_threshold: float = 0.15,
    batch_size: int = 256,
    enabled: bool = False,
) -> Tuple[List[float], List[Dict[str, Any]], List[Trade]]:
    """Compatibility wrapper that can stream windows before simulation.

    When disabled, falls back to :func:`simulate` exactly.
    """
    if not enabled:
        return simulate(
            ohlcv=ohlcv,
            composite_signal=composite_signal,
            symbol=symbol,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            signal_threshold=signal_threshold,
        )

    loader = StreamingDataLoader(
        data=np.arange(len(ohlcv), dtype=np.int32),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    # Materialise in streaming order to ensure backtest path can run without
    # loading additional transformed windows elsewhere.
    ordered_idx = np.concatenate([batch for batch in loader]) if len(ohlcv) else np.array([], dtype=np.int32)
    streamed_ohlcv = ohlcv.iloc[ordered_idx] if len(ordered_idx) else ohlcv.iloc[:0]
    streamed_signal = composite_signal.reindex(streamed_ohlcv.index)
    return simulate(
        ohlcv=streamed_ohlcv,
        composite_signal=streamed_signal,
        symbol=symbol,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        signal_threshold=signal_threshold,
    )
