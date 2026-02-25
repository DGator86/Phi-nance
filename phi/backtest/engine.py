"""
phi.backtest.engine — Vectorized Backtest Engine
==================================================
Pure-Python / NumPy backtest engine that takes:
  - OHLCV DataFrame
  - Blended signal Series (values in [-1, +1])
  - BacktestConfig dict

And returns a BacktestResult with:
  - equity_curve: pd.Series (portfolio value over time)
  - trades:       pd.DataFrame
  - metrics:      dict

No Lumibot dependency — runs in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    equity_curve:  pd.Series           # indexed like ohlcv
    trades:        pd.DataFrame        # one row per completed trade
    metrics:       Dict[str, Any]      # key metrics
    signals:       pd.Series           # raw signal series
    positions:     pd.Series           # position series (+1, -1, 0)
    start_capital: float
    end_capital:   float
    run_log:       List[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.end_capital - self.start_capital

    @property
    def net_pnl_pct(self) -> float:
        return (self.end_capital / self.start_capital - 1.0) if self.start_capital > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    bars_per_year: float = 252,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics."""

    if len(equity) < 2:
        return {}

    returns = equity.pct_change().dropna()
    end_val = float(equity.iloc[-1])
    total_ret = end_val / initial_capital - 1.0

    # CAGR
    n_bars = len(equity)
    n_years = n_bars / bars_per_year
    cagr = (end_val / initial_capital) ** (1.0 / max(n_years, 0.01)) - 1.0 if end_val > 0 else -1.0

    # Sharpe
    daily_ret_std = float(returns.std())
    daily_ret_mean = float(returns.mean())
    sharpe = (daily_ret_mean / (daily_ret_std + 1e-10)) * math.sqrt(bars_per_year) if daily_ret_std > 0 else 0.0

    # Sortino
    neg_returns = returns[returns < 0]
    downside_std = float(neg_returns.std()) if len(neg_returns) > 1 else 1e-10
    sortino = (daily_ret_mean / (downside_std + 1e-10)) * math.sqrt(bars_per_year)

    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / (rolling_max + 1e-10)
    max_dd = float(drawdown.min())

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0.0

    # Trade stats
    n_trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    largest_win = 0.0
    largest_loss = 0.0
    avg_hold = 0.0
    direction_accuracy = 0.0

    if not trades.empty and "pnl" in trades.columns:
        pnls = trades["pnl"].dropna()
        n_trades = len(pnls)
        if n_trades > 0:
            wins  = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            win_rate = len(wins) / n_trades
            total_wins  = float(wins.sum())
            total_losses = abs(float(losses.sum()))
            profit_factor = total_wins / (total_losses + 1e-10)
            avg_win  = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            largest_win  = float(wins.max()) if len(wins) > 0 else 0.0
            largest_loss = float(losses.min()) if len(losses) > 0 else 0.0

        if "bars_held" in trades.columns:
            avg_hold = float(trades["bars_held"].mean())

        # Direction accuracy: did we trade in the right direction?
        if "direction_correct" in trades.columns:
            direction_accuracy = float(trades["direction_correct"].mean())

    return {
        "total_return":       round(total_ret, 6),
        "cagr":               round(cagr, 6),
        "sharpe":             round(sharpe, 4),
        "sortino":            round(sortino, 4),
        "calmar":             round(calmar, 4),
        "max_drawdown":       round(max_dd, 6),
        "profit_factor":      round(profit_factor, 4),
        "win_rate":           round(win_rate, 4),
        "n_trades":           n_trades,
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "largest_win":        round(largest_win, 2),
        "largest_loss":       round(largest_loss, 2),
        "avg_hold_bars":      round(avg_hold, 1),
        "direction_accuracy": round(direction_accuracy, 4),
        "volatility_annual":  round(daily_ret_std * math.sqrt(bars_per_year), 6),
        "end_capital":        round(end_val, 2),
        "initial_capital":    round(initial_capital, 2),
        "net_pnl":            round(end_val - initial_capital, 2),
        "net_pnl_pct":        round(total_ret, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bars-per-year by timeframe
# ─────────────────────────────────────────────────────────────────────────────

_BARS_PER_YEAR: Dict[str, float] = {
    "1D":  252,
    "4H":  252 * 6.5,    # ~6.5 4H bars per day
    "1H":  252 * 6.5,
    "15m": 252 * 26,
    "5m":  252 * 78,
    "1m":  252 * 390,
}


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorized backtest engine.

    Parameters (config dict keys)
    -----------------------------
    initial_capital   : float
    signal_threshold  : float (min |signal| to enter)
    allow_short       : bool
    position_pct      : float (fraction of portfolio per trade)
    stop_loss_pct     : float or None
    take_profit_pct   : float or None
    trailing_stop_pct : float or None
    time_exit_bars    : int or None
    signal_exit       : bool (exit on opposing signal)
    timeframe         : str (for bars_per_year)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.initial_capital   = float(cfg.get("initial_capital",   100_000.0))
        self.signal_threshold  = float(cfg.get("signal_threshold",  0.10))
        self.allow_short       = bool(cfg.get("allow_short",         False))
        self.position_pct      = float(cfg.get("position_pct",       0.95))
        self.stop_loss_pct     = cfg.get("stop_loss_pct",     None)
        self.take_profit_pct   = cfg.get("take_profit_pct",   None)
        self.trailing_stop_pct = cfg.get("trailing_stop_pct", None)
        self.time_exit_bars    = cfg.get("time_exit_bars",    None)
        self.signal_exit       = bool(cfg.get("signal_exit",         True))
        self.timeframe         = str(cfg.get("timeframe",             "1D"))

    def run(
        self,
        ohlcv: pd.DataFrame,
        signal: pd.Series,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters
        ----------
        ohlcv             : OHLCV DataFrame
        signal            : signal series (same or compatible index), values in [-1, +1]
        progress_callback : optional callable(step: str, pct: float)

        Returns
        -------
        BacktestResult
        """
        run_log = []

        def _log(msg: str):
            run_log.append(msg)

        def _progress(step: str, pct: float):
            if progress_callback:
                progress_callback(step, pct)

        _progress("Aligning data...", 0.05)

        # ── Align signal to ohlcv index ──────────────────────────────────────
        signal = signal.reindex(ohlcv.index).ffill().fillna(0.0)

        close = ohlcv["close"].astype(float).values
        high  = ohlcv["high"].astype(float).values
        low   = ohlcv["low"].astype(float).values
        n     = len(close)

        _log(f"Bars: {n}  |  Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
        _progress("Generating positions...", 0.15)

        # ── Generate raw position series ─────────────────────────────────────
        raw_pos = np.zeros(n, dtype=float)
        for i in range(n):
            s = signal.iloc[i]
            if s > self.signal_threshold:
                raw_pos[i] = 1.0
            elif s < -self.signal_threshold and self.allow_short:
                raw_pos[i] = -1.0
            else:
                raw_pos[i] = 0.0

        _progress("Simulating trades...", 0.30)

        # ── Bar-by-bar simulation ─────────────────────────────────────────────
        equity       = np.full(n, self.initial_capital)
        cash         = self.initial_capital
        shares       = 0.0          # positive = long, negative = short
        entry_price  = 0.0
        entry_bar    = -1
        peak_price   = 0.0          # for trailing stop

        trades_list: List[Dict[str, Any]] = []

        for i in range(1, n):
            price = close[i]
            bar_high = high[i]
            bar_low  = low[i]

            in_trade = shares != 0.0

            # ── Stop checks (if in trade) ────────────────────────────────────
            if in_trade:
                is_long = shares > 0

                # Update trailing peak
                if self.trailing_stop_pct is not None:
                    if is_long:
                        peak_price = max(peak_price, bar_high)
                    else:
                        peak_price = min(peak_price, bar_low)

                exit_price: Optional[float] = None
                exit_reason: str = ""

                # Stop loss check
                if self.stop_loss_pct is not None:
                    if is_long:
                        sl_level = entry_price * (1.0 - self.stop_loss_pct)
                        if bar_low <= sl_level:
                            exit_price = sl_level
                            exit_reason = "stop_loss"
                    else:
                        sl_level = entry_price * (1.0 + self.stop_loss_pct)
                        if bar_high >= sl_level:
                            exit_price = sl_level
                            exit_reason = "stop_loss"

                # Take profit check
                if exit_price is None and self.take_profit_pct is not None:
                    if is_long:
                        tp_level = entry_price * (1.0 + self.take_profit_pct)
                        if bar_high >= tp_level:
                            exit_price = tp_level
                            exit_reason = "take_profit"
                    else:
                        tp_level = entry_price * (1.0 - self.take_profit_pct)
                        if bar_low <= tp_level:
                            exit_price = tp_level
                            exit_reason = "take_profit"

                # Trailing stop check
                if exit_price is None and self.trailing_stop_pct is not None:
                    if is_long:
                        ts_level = peak_price * (1.0 - self.trailing_stop_pct)
                        if bar_low <= ts_level:
                            exit_price = ts_level
                            exit_reason = "trailing_stop"
                    else:
                        ts_level = peak_price * (1.0 + self.trailing_stop_pct)
                        if bar_high >= ts_level:
                            exit_price = ts_level
                            exit_reason = "trailing_stop"

                # Time exit
                if exit_price is None and self.time_exit_bars is not None:
                    if (i - entry_bar) >= self.time_exit_bars:
                        exit_price = price
                        exit_reason = "time_exit"

                # Signal exit
                if exit_price is None and self.signal_exit:
                    new_pos = raw_pos[i]
                    if is_long and new_pos <= 0:
                        exit_price = price
                        exit_reason = "signal_exit"
                    elif not is_long and new_pos >= 0:
                        exit_price = price
                        exit_reason = "signal_exit"

                # Execute exit
                if exit_price is not None:
                    pnl = shares * (exit_price - entry_price)
                    cash += shares * exit_price if shares > 0 else -shares * (2 * entry_price - exit_price)
                    cash += abs(shares) * entry_price  # recover initial cost basis
                    cash += pnl
                    direction_correct = (pnl > 0)

                    trades_list.append({
                        "entry_bar":   entry_bar,
                        "exit_bar":    i,
                        "bars_held":   i - entry_bar,
                        "direction":   "long" if shares > 0 else "short",
                        "entry_date":  str(ohlcv.index[entry_bar])[:10],
                        "exit_date":   str(ohlcv.index[i])[:10],
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_price, 4),
                        "shares":      round(abs(shares), 4),
                        "pnl":         round(pnl, 2),
                        "pnl_pct":     round(pnl / (abs(shares) * entry_price + 1e-10), 6),
                        "exit_reason": exit_reason,
                        "direction_correct": direction_correct,
                    })

                    shares = 0.0
                    entry_price = 0.0
                    entry_bar   = -1
                    peak_price  = 0.0
                    in_trade    = False

            # ── Enter new trade ───────────────────────────────────────────────
            if not in_trade:
                new_pos = raw_pos[i]
                if new_pos != 0.0 and cash > price:
                    trade_cash = cash * self.position_pct
                    new_shares = (trade_cash / price) * new_pos
                    cost = abs(new_shares) * price
                    if cost <= cash:
                        cash    -= cost
                        shares   = new_shares
                        entry_price = price
                        entry_bar   = i
                        peak_price  = price

            # ── Update equity ─────────────────────────────────────────────────
            if shares != 0.0:
                position_value = shares * price if shares > 0 else -shares * (2 * entry_price - price)
                equity[i] = cash + position_value + abs(shares) * entry_price
            else:
                equity[i] = cash

            if i % max(n // 20, 1) == 0:
                _progress("Simulating trades...", 0.30 + 0.50 * (i / n))

        _progress("Computing metrics...", 0.85)

        # ── Close any open position at end ────────────────────────────────────
        if shares != 0.0:
            exit_price = close[-1]
            pnl = shares * (exit_price - entry_price)
            cash += abs(shares) * entry_price + pnl
            trades_list.append({
                "entry_bar":   entry_bar,
                "exit_bar":    n - 1,
                "bars_held":   n - 1 - entry_bar,
                "direction":   "long" if shares > 0 else "short",
                "entry_date":  str(ohlcv.index[entry_bar])[:10],
                "exit_date":   str(ohlcv.index[-1])[:10],
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price, 4),
                "shares":      round(abs(shares), 4),
                "pnl":         round(pnl, 2),
                "pnl_pct":     round(pnl / (abs(shares) * entry_price + 1e-10), 6),
                "exit_reason": "end_of_data",
                "direction_correct": pnl > 0,
            })
            equity[-1] = cash

        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        equity_series = pd.Series(equity, index=ohlcv.index, name="portfolio_value")

        # Bars per year
        bpy = _BARS_PER_YEAR.get(self.timeframe, 252)

        metrics = _compute_metrics(
            equity_series, trades_df, self.initial_capital, bpy
        )
        end_cap = float(equity[-1])

        _log(f"Trades: {len(trades_list)}  |  End capital: ${end_cap:,.2f}")
        _progress("Done", 1.0)

        position_series = pd.Series(raw_pos, index=ohlcv.index, name="position")

        return BacktestResult(
            equity_curve  = equity_series,
            trades        = trades_df,
            metrics       = metrics,
            signals       = signal,
            positions     = position_series,
            start_capital = self.initial_capital,
            end_capital   = end_cap,
            run_log       = run_log,
        )
