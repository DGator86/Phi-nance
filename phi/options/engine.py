"""
Full Options Backtesting Engine
================================

Black-Scholes pricing with realized-vol IV estimation.

Supported strategies:
  long_call, long_put, covered_call, cash_secured_put,
  bull_call_spread, bear_put_spread, straddle, strangle, iron_condor

Entry modes:
  signal   — enter when composite indicator signal crosses threshold
  periodic — enter every N bars regardless of signal

Exit rules (any combination):
  profit_target  — close when unrealized P&L >= exit_profit_pct * allocated
  stop_loss      — close when unrealized P&L <= exit_stop_pct  * allocated
  dte_exit       — close when remaining DTE <= dte_exit threshold
  expiry         — let position expire; close at intrinsic value

Portfolio accounting:
  Each trade reserves `allocated_capital` from cash.
  At close: cash += allocated_capital + realized_pnl.
  Portfolio value = cash + sum(allocated + unrealized_pnl) for open trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> Tuple[float, float, float, float, float]:
    """
    Black-Scholes pricing.

    Parameters
    ----------
    S, K      : spot and strike prices
    T         : time to expiry in years
    r         : annualized risk-free rate (decimal)
    sigma     : annualized implied volatility (decimal)
    option_type : 'call' or 'put'

    Returns
    -------
    (price, delta, gamma, theta_daily, vega_per_1pct)
    """
    sigma = max(sigma, 0.005)
    if T <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            intrinsic = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        else:
            intrinsic = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-r * T)
    npdf_d1 = _npdf(d1)

    gamma = npdf_d1 / (S * sigma * sqrtT)
    vega = S * npdf_d1 * sqrtT * 0.01  # per 1% vol move

    if option_type == "call":
        price = S * _ncdf(d1) - K * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = (-(S * npdf_d1 * sigma) / (2 * sqrtT) - r * K * disc * _ncdf(d2)) / 365.0
    else:
        price = K * disc * _ncdf(-d2) - S * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0
        theta = (-(S * npdf_d1 * sigma) / (2 * sqrtT) + r * K * disc * _ncdf(-d2)) / 365.0

    return max(0.0, price), delta, gamma, theta, vega


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsLeg:
    option_type: str    # 'call' | 'put' | 'share'
    action: str         # 'buy'  | 'sell'
    strike: float
    dte_at_entry: int   # 0 for share legs
    entry_premium: float
    entry_delta: float
    quantity: int = 1   # contracts; share leg = 100 shares

    @property
    def sign(self) -> float:
        return 1.0 if self.action == "buy" else -1.0


@dataclass
class ActiveTrade:
    strategy_type: str
    legs: List[OptionsLeg]
    entry_bar: int
    entry_date: Any
    spot_at_entry: float
    net_entry_cost: float    # net debit (+) or credit (−) at entry
    allocated_capital: float # max capital at risk (margin / premium paid)
    entry_iv: float


@dataclass
class ClosedTrade:
    strategy_type: str
    entry_date: Any
    exit_date: Any
    entry_bar: int
    exit_bar: int
    spot_at_entry: float
    spot_at_exit: float
    net_entry_cost: float
    net_exit_value: float
    pnl: float             # net_exit_value − net_entry_cost (+ = profit)
    pnl_pct: float         # pnl / allocated_capital
    close_reason: str      # profit_target | stop_loss | dte_exit | expiry | end_of_data
    entry_iv: float
    exit_iv: float
    allocated_capital: float
    entry_delta: float     # net option delta at entry (for display)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy catalog
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_NAMES: Dict[str, str] = {
    "long_call":        "Long Call",
    "long_put":         "Long Put",
    "covered_call":     "Covered Call",
    "cash_secured_put": "Cash-Secured Put",
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread":  "Bear Put Spread",
    "straddle":         "Straddle",
    "strangle":         "Strangle",
    "iron_condor":      "Iron Condor",
}

# Which direction of the underlying indicator signal triggers an entry
ENTRY_BIAS: Dict[str, str] = {
    "long_call":        "bullish",
    "long_put":         "bearish",
    "covered_call":     "neutral",
    "cash_secured_put": "neutral",
    "bull_call_spread": "bullish",
    "bear_put_spread":  "bearish",
    "straddle":         "volatile",
    "strangle":         "volatile",
    "iron_condor":      "neutral",
}


# ─────────────────────────────────────────────────────────────────────────────
# Leg builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_legs(
    strategy_type: str,
    spot: float,
    iv: float,
    dte: int,
    r: float,
    otm_pct: float,
) -> List[OptionsLeg]:
    """Construct legs with Black-Scholes entry premiums."""
    T = dte / 365.0
    atm = round(spot)
    otm_c = round(spot * (1.0 + otm_pct))
    otm_p = round(spot * (1.0 - otm_pct))
    far_c = round(spot * (1.0 + 2.0 * otm_pct))
    far_p = round(spot * (1.0 - 2.0 * otm_pct))

    def _opt(otype: str, action: str, k: float) -> OptionsLeg:
        p, d, _, _, _ = black_scholes(spot, k, T, r, iv, otype)
        return OptionsLeg(option_type=otype, action=action, strike=k,
                          dte_at_entry=dte, entry_premium=p, entry_delta=d)

    def _share() -> OptionsLeg:
        return OptionsLeg(option_type="share", action="buy", strike=0.0,
                          dte_at_entry=0, entry_premium=spot, entry_delta=1.0)

    builder: Dict[str, List[OptionsLeg]] = {
        "long_call":        [_opt("call", "buy",  atm)],
        "long_put":         [_opt("put",  "buy",  atm)],
        "covered_call":     [_share(), _opt("call", "sell", otm_c)],
        "cash_secured_put": [_opt("put",  "sell", otm_p)],
        "bull_call_spread": [_opt("call", "buy",  atm),   _opt("call", "sell", otm_c)],
        "bear_put_spread":  [_opt("put",  "buy",  atm),   _opt("put",  "sell", otm_p)],
        "straddle":         [_opt("call", "buy",  atm),   _opt("put",  "buy",  atm)],
        "strangle":         [_opt("call", "buy",  otm_c), _opt("put",  "buy",  otm_p)],
        "iron_condor": [
            _opt("put",  "sell", otm_p),
            _opt("put",  "buy",  far_p),
            _opt("call", "sell", otm_c),
            _opt("call", "buy",  far_c),
        ],
    }
    return builder.get(strategy_type, [])


def _net_entry_cost(legs: List[OptionsLeg]) -> float:
    """Net cash flow at entry (positive = net debit, negative = net credit)."""
    total = 0.0
    for leg in legs:
        if leg.option_type == "share":
            total += leg.sign * leg.entry_premium * 100.0
        else:
            total += leg.sign * leg.entry_premium * 100.0
    return total


def _allocated_capital(
    strategy_type: str,
    legs: List[OptionsLeg],
    spot: float,
    otm_pct: float,
) -> float:
    """Capital reserved per trade (= max loss for defined-risk strategies)."""
    net = _net_entry_cost(legs)

    if strategy_type in ("long_call", "long_put", "straddle", "strangle", "bull_call_spread", "bear_put_spread"):
        # Debit strategies: max loss = net debit paid
        return max(net, 1.0)

    if strategy_type == "covered_call":
        # Share cost − call premium received
        share_cost = spot * 100.0
        call_credit = sum(
            abs(leg.entry_premium) * 100.0
            for leg in legs if leg.option_type == "call" and leg.action == "sell"
        )
        return max(share_cost - call_credit, 1.0)

    if strategy_type == "cash_secured_put":
        # Cash reserved = put strike * 100 − premium received
        for leg in legs:
            if leg.option_type == "put" and leg.action == "sell":
                return max(leg.strike * 100.0 - leg.entry_premium * 100.0, 1.0)

    if strategy_type == "iron_condor":
        # Max loss = spread width − net credit
        spread_width = spot * otm_pct * 100.0
        return max(spread_width + net, 1.0)  # net is negative (credit)

    return max(abs(net), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Pricing at current bar
# ─────────────────────────────────────────────────────────────────────────────

def _leg_current_value(leg: OptionsLeg, spot: float, remaining_T: float, r: float, iv: float) -> float:
    """Theoretical value of one leg at current market conditions."""
    if leg.option_type == "share":
        return spot * 100.0
    p, _, _, _, _ = black_scholes(spot, leg.strike, remaining_T, r, iv, leg.option_type)
    return p


def _trade_pnl(
    trade: ActiveTrade, spot: float, bar_idx: int, r: float, iv: float
) -> Tuple[float, float]:
    """
    Returns (unrealized_pnl, current_net_value) for an open trade.

    unrealized_pnl = current_net_value − net_entry_cost
    """
    elapsed = bar_idx - trade.entry_bar
    # Use the first option leg's dte_at_entry (shares have 0)
    first_dte = next(
        (leg.dte_at_entry for leg in trade.legs if leg.option_type != "share"), 30
    )
    remaining_dte = max(0, first_dte - elapsed)
    remaining_T = remaining_dte / 365.0

    current_net = 0.0
    for leg in trade.legs:
        cv = _leg_current_value(leg, spot, remaining_T, r, iv)
        if leg.option_type == "share":
            current_net += leg.sign * cv          # cv = spot * 100
        else:
            current_net += leg.sign * cv * 100.0  # cv = price per share; ×100 for contract

    pnl = current_net - trade.net_entry_cost
    return pnl, current_net


# ─────────────────────────────────────────────────────────────────────────────
# Realized volatility helper
# ─────────────────────────────────────────────────────────────────────────────

def _realized_vol(log_rets: np.ndarray, bar_idx: int, lookback: int) -> float:
    """Annualized realized vol from the last `lookback` log-returns."""
    start = max(0, bar_idx - lookback)
    window = log_rets[start:bar_idx]
    if len(window) < 3:
        return 0.20
    return max(float(np.std(window) * math.sqrt(252)), 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_options_backtest_full(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    strategy_type: str = "long_call",
    entry_mode: str = "signal",            # 'signal' | 'periodic'
    entry_signal: Optional[pd.Series] = None,
    signal_threshold: float = 0.15,
    dte: int = 30,
    iv_factor: float = 1.0,
    iv_lookback: int = 21,
    position_pct: float = 0.05,
    otm_pct: float = 0.05,
    exit_profit_pct: float = 0.50,
    exit_stop_pct: float = 0.50,           # positive — stored as fraction of alloc
    exit_dte: int = 5,
    hold_to_expiry: bool = False,
    max_open_trades: int = 3,
    risk_free_rate: float = 0.045,
    initial_capital: float = 100_000.0,
    periodic_entry_days: int = 21,
) -> Dict:
    """
    Full options backtest on a cached OHLCV DataFrame.

    Returns
    -------
    dict with keys:
      portfolio_value, total_return, cagr, max_drawdown, sharpe,
      total_trades, win_rate, avg_win, avg_loss, profit_factor,
      trade_log (DataFrame), pnls (list), iv_series (list)
    """
    # ── Normalize columns ────────────────────────────────────────────────────
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return _empty_result(initial_capital)

    df = df.sort_index()
    closes = df["close"].values.astype(float)
    n = len(closes)

    if n < iv_lookback + 5:
        return _empty_result(initial_capital)

    # Pre-compute log-returns for IV estimation
    log_rets = np.log(np.maximum(closes[1:], 1e-8) / np.maximum(closes[:-1], 1e-8))
    dates = list(df.index)

    # ── Align entry signal ───────────────────────────────────────────────────
    sig_arr = np.zeros(n, dtype=float)
    if entry_signal is not None and len(entry_signal) > 0:
        try:
            aligned = entry_signal.reindex(df.index).ffill().bfill().fillna(0)
            sig_arr = aligned.values.astype(float)[:n]
        except Exception:
            pass

    bias = ENTRY_BIAS.get(strategy_type, "neutral")

    def _entry_ok(bar: int, last_bar: int) -> bool:
        if entry_mode == "periodic":
            return (bar - last_bar) >= periodic_entry_days
        sig = float(sig_arr[bar]) if bar < len(sig_arr) else 0.0
        if bias == "bullish":
            return sig > signal_threshold
        if bias == "bearish":
            return sig < -signal_threshold
        if bias == "volatile":
            return abs(sig) > signal_threshold
        # neutral / iron_condor: enter when signal is quiet
        return abs(sig) <= signal_threshold

    def _should_exit(trade: ActiveTrade, pnl: float, bar: int) -> Optional[str]:
        if hold_to_expiry:
            elapsed = bar - trade.entry_bar
            first_dte = next(
                (leg.dte_at_entry for leg in trade.legs if leg.option_type != "share"), 30
            )
            if elapsed >= first_dte:
                return "expiry"
            return None
        alloc = max(trade.allocated_capital, 1.0)
        if pnl >= exit_profit_pct * alloc:
            return "profit_target"
        if pnl <= -abs(exit_stop_pct) * alloc:
            return "stop_loss"
        elapsed = bar - trade.entry_bar
        first_dte = next(
            (leg.dte_at_entry for leg in trade.legs if leg.option_type != "share"), 30
        )
        if (first_dte - elapsed) <= exit_dte:
            return "dte_exit"
        return None

    # ── Backtest loop ─────────────────────────────────────────────────────────
    capital = float(initial_capital)
    open_trades: List[ActiveTrade] = []
    closed_trades: List[ClosedTrade] = []
    portfolio_values: List[float] = []
    iv_series: List[float] = []
    last_entry_bar = -9_999

    for bar in range(n):
        spot = float(closes[bar])
        iv = _realized_vol(log_rets, bar, iv_lookback) * iv_factor
        r = risk_free_rate
        iv_series.append(iv)

        # ── 1. Check exits ───────────────────────────────────────────────────
        to_close: List[Tuple[int, str]] = []
        for i, trade in enumerate(open_trades):
            pnl, _ = _trade_pnl(trade, spot, bar, r, iv)
            reason = _should_exit(trade, pnl, bar)
            if reason:
                to_close.append((i, reason))

        for idx, reason in reversed(to_close):
            trade = open_trades[idx]
            exit_iv = iv
            elapsed = bar - trade.entry_bar
            first_dte = next(
                (leg.dte_at_entry for leg in trade.legs if leg.option_type != "share"), 30
            )
            remaining_dte = max(0, first_dte - elapsed)
            remaining_T = remaining_dte / 365.0

            exit_net = 0.0
            for leg in trade.legs:
                cv = _leg_current_value(leg, spot, remaining_T, r, exit_iv)
                if leg.option_type == "share":
                    exit_net += leg.sign * cv
                else:
                    exit_net += leg.sign * cv * 100.0

            realized_pnl = exit_net - trade.net_entry_cost
            capital += trade.allocated_capital + realized_pnl

            net_delta = sum(
                leg.sign * leg.entry_delta
                for leg in trade.legs if leg.option_type != "share"
            )
            closed_trades.append(ClosedTrade(
                strategy_type=trade.strategy_type,
                entry_date=trade.entry_date,
                exit_date=dates[bar] if bar < len(dates) else None,
                entry_bar=trade.entry_bar,
                exit_bar=bar,
                spot_at_entry=trade.spot_at_entry,
                spot_at_exit=spot,
                net_entry_cost=trade.net_entry_cost,
                net_exit_value=exit_net,
                pnl=realized_pnl,
                pnl_pct=realized_pnl / max(trade.allocated_capital, 1.0),
                close_reason=reason,
                entry_iv=trade.entry_iv,
                exit_iv=exit_iv,
                allocated_capital=trade.allocated_capital,
                entry_delta=net_delta,
            ))
            open_trades.pop(idx)

        # ── 2. Entry ─────────────────────────────────────────────────────────
        if len(open_trades) < max_open_trades and _entry_ok(bar, last_entry_bar):
            legs = _build_legs(strategy_type, spot, iv, dte, r, otm_pct)
            if legs:
                net_cost = _net_entry_cost(legs)
                alloc = _allocated_capital(strategy_type, legs, spot, otm_pct)
                budget = capital * position_pct
                if alloc > 0 and capital >= alloc and budget >= alloc * 0.5:
                    capital -= alloc
                    open_trades.append(ActiveTrade(
                        strategy_type=strategy_type,
                        legs=legs,
                        entry_bar=bar,
                        entry_date=dates[bar] if bar < len(dates) else None,
                        spot_at_entry=spot,
                        net_entry_cost=net_cost,
                        allocated_capital=alloc,
                        entry_iv=iv,
                    ))
                    last_entry_bar = bar

        # ── 3. Mark-to-market ────────────────────────────────────────────────
        unrealized = 0.0
        open_alloc = 0.0
        for trade in open_trades:
            pnl, _ = _trade_pnl(trade, spot, bar, r, iv)
            unrealized += pnl
            open_alloc += trade.allocated_capital
        pv = max(capital + open_alloc + unrealized, 0.0)
        portfolio_values.append(pv)

    # ── Close remaining open trades at last bar ───────────────────────────────
    if open_trades:
        spot = float(closes[-1])
        iv = _realized_vol(log_rets, n - 1, iv_lookback) * iv_factor
        r = risk_free_rate
        for trade in open_trades:
            elapsed = (n - 1) - trade.entry_bar
            first_dte = next(
                (leg.dte_at_entry for leg in trade.legs if leg.option_type != "share"), 30
            )
            remaining_T = max(0, first_dte - elapsed) / 365.0
            exit_net = 0.0
            for leg in trade.legs:
                cv = _leg_current_value(leg, spot, remaining_T, r, iv)
                if leg.option_type == "share":
                    exit_net += leg.sign * cv
                else:
                    exit_net += leg.sign * cv * 100.0
            realized_pnl = exit_net - trade.net_entry_cost
            net_delta = sum(
                leg.sign * leg.entry_delta
                for leg in trade.legs if leg.option_type != "share"
            )
            closed_trades.append(ClosedTrade(
                strategy_type=trade.strategy_type,
                entry_date=trade.entry_date,
                exit_date=dates[-1] if dates else None,
                entry_bar=trade.entry_bar,
                exit_bar=n - 1,
                spot_at_entry=trade.spot_at_entry,
                spot_at_exit=spot,
                net_entry_cost=trade.net_entry_cost,
                net_exit_value=exit_net,
                pnl=realized_pnl,
                pnl_pct=realized_pnl / max(trade.allocated_capital, 1.0),
                close_reason="end_of_data",
                entry_iv=trade.entry_iv,
                exit_iv=iv,
                allocated_capital=trade.allocated_capital,
                entry_delta=net_delta,
            ))

    # ── Metrics ───────────────────────────────────────────────────────────────
    pv_arr = np.array(portfolio_values, dtype=float)
    if len(pv_arr) == 0:
        return _empty_result(initial_capital)

    total_return = (pv_arr[-1] / initial_capital - 1.0) if initial_capital else 0.0
    years = max(len(df) / 252.0, 0.1)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0

    peak = np.maximum.accumulate(pv_arr)
    dd = (pv_arr - peak) / np.maximum(peak, 1e-8)
    max_dd = float(np.min(dd))

    pv_ret = np.diff(pv_arr) / np.maximum(pv_arr[:-1], 1e-8)
    sharpe = (
        float(np.mean(pv_ret) / np.std(pv_ret) * np.sqrt(252))
        if len(pv_ret) > 1 and np.std(pv_ret) > 0
        else 0.0
    )

    pnls = [ct.pnl for ct in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    max_consec_loss = _max_consecutive_losses(pnls)

    # ── Trade log DataFrame ───────────────────────────────────────────────────
    if closed_trades:
        trade_log = pd.DataFrame([{
            "Entry Date":    _fmt_date(ct.entry_date),
            "Exit Date":     _fmt_date(ct.exit_date),
            "Strategy":      STRATEGY_NAMES.get(ct.strategy_type, ct.strategy_type),
            "Spot Entry":    round(ct.spot_at_entry, 2),
            "Spot Exit":     round(ct.spot_at_exit, 2),
            "Entry IV":      f"{ct.entry_iv:.1%}",
            "Exit IV":       f"{ct.exit_iv:.1%}",
            "Net δ Entry":   round(ct.entry_delta, 3),
            "Cost ($)":      round(ct.net_entry_cost, 2),
            "P&L ($)":       round(ct.pnl, 2),
            "P&L %":         f"{ct.pnl_pct:+.1%}",
            "Hold (bars)":   ct.exit_bar - ct.entry_bar,
            "Exit Reason":   ct.close_reason,
        } for ct in closed_trades])
    else:
        trade_log = pd.DataFrame()

    return {
        "portfolio_value": list(pv_arr),
        "total_return":    total_return,
        "cagr":            cagr,
        "max_drawdown":    max_dd,
        "sharpe":          sharpe,
        "total_trades":    len(closed_trades),
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "profit_factor":   profit_factor,
        "max_consec_loss": max_consec_loss,
        "gross_win":       gross_win,
        "gross_loss":      gross_loss,
        "trade_log":       trade_log,
        "pnls":            pnls,
        "iv_series":       iv_series,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_consecutive_losses(pnls: List[float]) -> int:
    max_run = run = 0
    for p in pnls:
        if p < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _fmt_date(d: Any) -> str:
    if d is None:
        return ""
    try:
        return str(d)[:10]
    except Exception:
        return str(d)


def _empty_result(capital: float) -> Dict:
    return {
        "portfolio_value": [capital],
        "total_return":    0.0,
        "cagr":            0.0,
        "max_drawdown":    0.0,
        "sharpe":          0.0,
        "total_trades":    0,
        "win_rate":        0.0,
        "avg_win":         0.0,
        "avg_loss":        0.0,
        "profit_factor":   0.0,
        "max_consec_loss": 0,
        "gross_win":       0.0,
        "gross_loss":      0.0,
        "trade_log":       pd.DataFrame(),
        "pnls":            [],
        "iv_series":       [],
    }
