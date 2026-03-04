"""
phi.options.simulator — Options Backtest Simulator
====================================================
Simplified options backtest using Black-Scholes approximations.
No live chain data required — simulates P&L from OHLCV.

Supported structures:
  long_call, long_put, debit_spread (bull_call / bear_put)

Methodology:
  - At signal entry: "buy" an ATM option at Black-Scholes fair value
  - Compute approximate option price using simplified BSM
  - Track daily theta decay
  - Exit on signal reversal, profit target, stop loss, or DTE expiry
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility (annualized)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window // 2).std() * math.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsBacktestResult:
    equity_curve: pd.Series
    trades:       pd.DataFrame
    metrics:      Dict[str, Any]
    start_capital: float
    end_capital:   float


# ─────────────────────────────────────────────────────────────────────────────
# Options Simulator
# ─────────────────────────────────────────────────────────────────────────────

class OptionsSimulator:
    """
    Simulates options trades driven by a signal series.

    Config keys
    -----------
    initial_capital   : float
    structure         : 'long_call' | 'long_put' | 'debit_spread'
    target_dte        : int (days to expiry at entry)
    iv_floor          : float (minimum IV to use, e.g. 0.15)
    profit_exit_pct   : float (exit when option gained this fraction, e.g. 0.50)
    stop_exit_pct     : float (exit when option lost this fraction, e.g. 1.00)
    position_pct      : float (fraction of capital per trade)
    contracts         : int (number of contracts per trade; 0 = auto from position_pct)
    risk_free_rate    : float (default 0.04)
    signal_threshold  : float
    signal_exit       : bool
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.initial_capital = float(cfg.get("initial_capital",  100_000.0))
        self.structure       = str(cfg.get("structure",          "long_call"))
        self.target_dte      = int(cfg.get("target_dte",         45))
        self.iv_floor        = float(cfg.get("iv_floor",          0.15))
        self.profit_exit_pct = float(cfg.get("profit_exit_pct",   0.50))
        self.stop_exit_pct   = float(cfg.get("stop_exit_pct",     1.00))
        self.position_pct    = float(cfg.get("position_pct",      0.05))   # 5% risk per trade
        self.contracts       = int(cfg.get("contracts",           0))
        self.rfr             = float(cfg.get("risk_free_rate",    0.04))
        self.signal_threshold = float(cfg.get("signal_threshold", 0.10))
        self.signal_exit     = bool(cfg.get("signal_exit",        True))
        self.spread_width_pct = float(cfg.get("spread_width_pct", 0.03))

    def run(
        self,
        ohlcv: pd.DataFrame,
        signal: pd.Series,
    ) -> OptionsBacktestResult:
        """Run the options backtest simulation."""

        close = ohlcv["close"].astype(float)
        vol   = realized_vol(close, 20).fillna(self.iv_floor).clip(lower=self.iv_floor)
        n     = len(close)

        signal = signal.reindex(ohlcv.index).ffill().fillna(0.0)

        cash     = self.initial_capital
        equity   = np.full(n, self.initial_capital)
        trades_list: List[Dict[str, Any]] = []

        # State
        in_trade    = False
        entry_bar   = -1
        entry_premium = 0.0
        is_call     = True
        n_contracts = 1
        strike      = 0.0
        entry_dte   = self.target_dte

        for i in range(1, n):
            S    = float(close.iloc[i])
            iv   = float(vol.iloc[i])
            sig  = float(signal.iloc[i])

            T    = max(entry_dte - (i - entry_bar), 0) / 365.0 if in_trade else 0.0

            # ── Check exit if in trade ────────────────────────────────────────
            if in_trade:
                # Current option value
                curr_val = self._option_value(S, strike, T, iv, is_call)
                pnl_per_contract = (curr_val - entry_premium) * 100

                # Profit / stop exit
                exit_reason: Optional[str] = None
                gain_pct = (curr_val - entry_premium) / (entry_premium + 1e-10)
                loss_pct = -(curr_val - entry_premium) / (entry_premium + 1e-10)

                if gain_pct >= self.profit_exit_pct:
                    exit_reason = "profit_target"
                elif loss_pct >= self.stop_exit_pct:
                    exit_reason = "stop_loss"
                elif (i - entry_bar) >= entry_dte:
                    exit_reason = "expiry"
                elif self.signal_exit:
                    if is_call and sig < -self.signal_threshold:
                        exit_reason = "signal_exit"
                    elif not is_call and sig > self.signal_threshold:
                        exit_reason = "signal_exit"

                if exit_reason:
                    pnl = pnl_per_contract * n_contracts
                    cash += n_contracts * curr_val * 100
                    trades_list.append({
                        "entry_bar":    entry_bar,
                        "exit_bar":     i,
                        "bars_held":    i - entry_bar,
                        "structure":    self.structure,
                        "direction":    "call" if is_call else "put",
                        "entry_date":   str(ohlcv.index[entry_bar])[:10],
                        "exit_date":    str(ohlcv.index[i])[:10],
                        "entry_price":  round(S, 2),
                        "exit_price":   round(S, 2),
                        "strike":       round(strike, 2),
                        "entry_prem":   round(entry_premium, 4),
                        "exit_prem":    round(curr_val, 4),
                        "n_contracts":  n_contracts,
                        "pnl":          round(pnl, 2),
                        "pnl_pct":      round(gain_pct if gain_pct > 0 else -loss_pct, 6),
                        "exit_reason":  exit_reason,
                        "iv_at_entry":  round(iv, 4),
                        "direction_correct": pnl > 0,
                    })
                    in_trade = False
                    entry_bar = -1

            # ── Enter new trade ───────────────────────────────────────────────
            if not in_trade and abs(sig) > self.signal_threshold:
                is_call      = sig > 0
                strike       = S  # ATM
                entry_dte    = self.target_dte
                T_entry      = entry_dte / 365.0

                prem = self._option_value(S, strike, T_entry, iv, is_call)

                if prem <= 0 or prem > S * 0.5:
                    continue

                # Position sizing
                if self.contracts > 0:
                    n_contracts = self.contracts
                else:
                    risk_per_trade = cash * self.position_pct
                    n_contracts = max(1, int(risk_per_trade / (prem * 100)))

                cost = n_contracts * prem * 100
                if cost > cash:
                    n_contracts = max(1, int(cash / (prem * 100)))
                    cost = n_contracts * prem * 100

                if cost > cash:
                    continue

                cash          -= cost
                in_trade       = True
                entry_bar      = i
                entry_premium  = prem

            # ── Update equity ─────────────────────────────────────────────────
            if in_trade:
                curr_T   = max(entry_dte - (i - entry_bar), 0) / 365.0
                curr_val = self._option_value(S, strike, curr_T, iv, is_call)
                equity[i] = cash + n_contracts * curr_val * 100
            else:
                equity[i] = cash

        # Close open trade at end
        if in_trade:
            S = float(close.iloc[-1])
            iv = float(vol.iloc[-1])
            curr_val = self._option_value(S, strike, 0.001, iv, is_call)
            pnl = (curr_val - entry_premium) * 100 * n_contracts
            cash += n_contracts * curr_val * 100
            trades_list.append({
                "entry_bar": entry_bar, "exit_bar": n - 1,
                "bars_held": n - 1 - entry_bar, "structure": self.structure,
                "direction": "call" if is_call else "put",
                "entry_date": str(ohlcv.index[entry_bar])[:10],
                "exit_date": str(ohlcv.index[-1])[:10],
                "entry_price": round(float(close.iloc[entry_bar]), 2),
                "exit_price": round(S, 2), "strike": round(strike, 2),
                "entry_prem": round(entry_premium, 4), "exit_prem": round(curr_val, 4),
                "n_contracts": n_contracts, "pnl": round(pnl, 2),
                "pnl_pct": round((curr_val - entry_premium) / (entry_premium + 1e-10), 6),
                "exit_reason": "end_of_data", "iv_at_entry": round(iv, 4),
                "direction_correct": pnl > 0,
            })
            equity[-1] = cash

        trades_df    = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        equity_s     = pd.Series(equity, index=ohlcv.index, name="portfolio_value")
        end_cap      = float(equity[-1])
        metrics      = self._compute_metrics(equity_s, trades_df)
        return OptionsBacktestResult(equity_s, trades_df, metrics, self.initial_capital, end_cap)

    def _option_value(self, S, K, T, iv, is_call) -> float:
        iv = max(iv, self.iv_floor)
        if is_call:
            if self.structure == "debit_spread":
                long_val  = bs_call(S, K, T, self.rfr, iv)
                short_K   = K * (1.0 + self.spread_width_pct)
                short_val = bs_call(S, short_K, T, self.rfr, iv)
                return long_val - short_val
            return bs_call(S, K, T, self.rfr, iv)
        else:
            if self.structure == "debit_spread":
                long_val  = bs_put(S, K, T, self.rfr, iv)
                short_K   = K * (1.0 - self.spread_width_pct)
                short_val = bs_put(S, short_K, T, self.rfr, iv)
                return long_val - short_val
            return bs_put(S, K, T, self.rfr, iv)

    def _compute_metrics(self, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
        import math
        rets = equity.pct_change().dropna()
        end  = float(equity.iloc[-1])
        ret  = end / self.initial_capital - 1.0
        std  = float(rets.std())
        mean = float(rets.mean())
        sharpe = (mean / (std + 1e-10)) * math.sqrt(252) if std > 0 else 0.0
        dd   = float(((equity - equity.cummax()) / equity.cummax()).min())
        n_tr = len(trades)
        wr   = float((trades["pnl"] > 0).mean()) if n_tr > 0 and "pnl" in trades.columns else 0.0
        wins = trades["pnl"][trades["pnl"] > 0].sum() if n_tr > 0 else 0
        loss = abs(trades["pnl"][trades["pnl"] < 0].sum()) if n_tr > 0 else 1e-10
        pf   = wins / (loss + 1e-10)
        return {
            "total_return": round(ret, 6), "sharpe": round(sharpe, 4),
            "max_drawdown": round(dd, 6), "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4), "n_trades": n_tr,
            "end_capital": round(end, 2), "initial_capital": self.initial_capital,
            "net_pnl": round(end - self.initial_capital, 2),
        }
