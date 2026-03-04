"""
phinance.backtest.portfolio
==============================

Portfolio-Level Backtester — multi-asset correlated backtest with
capital allocation and position sizing.

Architecture
------------
The portfolio backtester takes a dictionary of OHLCV DataFrames (one per
symbol) and a dictionary of pre-computed signal Series, then simulates a
portfolio that:

  1. Allocates capital across assets using a configurable allocation rule
     (equal-weight, risk-parity, fixed-weight).
  2. Applies position sizing (fixed-fraction or Kelly fraction).
  3. Deducts transaction costs on each entry/exit.
  4. Computes per-asset and aggregate equity curves.
  5. Reports portfolio-level risk metrics.

Public API
----------
  AllocationMethod       — enum: EQUAL, RISK_PARITY, FIXED
  PortfolioConfig        — configuration dataclass
  AssetResult            — per-asset backtest result
  PortfolioResult        — aggregate portfolio result
  PortfolioBacktester    — main controller
  run_portfolio_backtest — convenience function
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    total_return,
    cagr,
    win_rate as compute_win_rate,
)
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Config
# ─────────────────────────────────────────────────────────────────────────────


class AllocationMethod(str, Enum):
    """Capital allocation strategy across assets."""
    EQUAL       = "equal"        # equal weight to every asset
    RISK_PARITY = "risk_parity"  # inverse-volatility weighting
    FIXED       = "fixed"        # user-supplied weights dict


@dataclass
class PortfolioConfig:
    """
    Configuration for PortfolioBacktester.

    Attributes
    ----------
    initial_capital   : float  — starting capital (default 100 000)
    allocation        : AllocationMethod  — how to split capital
    fixed_weights     : dict   — {symbol: fraction} used when allocation=FIXED
    position_size     : float  — fraction of allocated capital per position (default 1.0)
    transaction_cost  : float  — round-trip cost fraction (default 0.001)
    signal_threshold  : float  — signal value above which we go long (default 0.1)
    allow_short       : bool   — if True, short on signal < -threshold (default False)
    rebalance_bars    : int    — rebalance allocation every N bars (default 0 = no rebalance)
    """

    initial_capital:  float = 100_000.0
    allocation:       AllocationMethod = AllocationMethod.EQUAL
    fixed_weights:    Dict[str, float] = field(default_factory=dict)
    position_size:    float = 1.0
    transaction_cost: float = 0.001
    signal_threshold: float = 0.1
    allow_short:      bool = False
    rebalance_bars:   int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["allocation"] = self.allocation.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AssetResult:
    """
    Per-asset backtest result within a portfolio.

    Attributes
    ----------
    symbol        : str
    allocation    : float — capital fraction allocated to this asset
    total_return  : float
    sharpe        : float
    max_drawdown  : float
    win_rate      : float
    num_trades    : int
    equity_curve  : np.ndarray
    """

    symbol:       str = ""
    allocation:   float = 0.0
    total_return: float = 0.0
    sharpe:       float = 0.0
    max_drawdown: float = 0.0
    win_rate:     float = 0.0
    num_trades:   int = 0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":       self.symbol,
            "allocation":   self.allocation,
            "total_return": self.total_return,
            "sharpe":       self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate":     self.win_rate,
            "num_trades":   self.num_trades,
        }

    def __repr__(self) -> str:
        return (
            f"AssetResult(symbol={self.symbol!r}, "
            f"return={self.total_return:.2%}, "
            f"sharpe={self.sharpe:.2f})"
        )


@dataclass
class PortfolioResult:
    """
    Aggregate portfolio backtest result.

    Attributes
    ----------
    portfolio_id      : str
    symbols           : list[str]
    initial_capital   : float
    final_capital     : float
    total_return      : float
    cagr              : float
    sharpe            : float
    sortino           : float
    max_drawdown      : float
    win_rate          : float
    num_trades        : int
    portfolio_equity  : np.ndarray — aggregate NAV curve
    asset_results     : dict       — {symbol: AssetResult}
    correlation_matrix: np.ndarray
    elapsed_ms        : float
    config            : dict
    """

    portfolio_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbols:           List[str] = field(default_factory=list)
    initial_capital:   float = 100_000.0
    final_capital:     float = 100_000.0
    total_return:      float = 0.0
    cagr:              float = 0.0
    sharpe:            float = 0.0
    sortino:           float = 0.0
    max_drawdown:      float = 0.0
    win_rate:          float = 0.0
    num_trades:        int = 0
    portfolio_equity:  np.ndarray = field(default_factory=lambda: np.array([]))
    asset_results:     Dict[str, AssetResult] = field(default_factory=dict)
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_ms:        float = 0.0
    config:            Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id":     self.portfolio_id,
            "symbols":          self.symbols,
            "initial_capital":  self.initial_capital,
            "final_capital":    self.final_capital,
            "total_return":     self.total_return,
            "cagr":             self.cagr,
            "sharpe":           self.sharpe,
            "sortino":          self.sortino,
            "max_drawdown":     self.max_drawdown,
            "win_rate":         self.win_rate,
            "num_trades":       self.num_trades,
            "asset_results":    {s: ar.to_dict() for s, ar in self.asset_results.items()},
        }

    def summary(self) -> str:
        lines = [
            f"Portfolio {self.portfolio_id}",
            f"  Assets:       {', '.join(self.symbols)}",
            f"  Total return: {self.total_return:.2%}",
            f"  CAGR:         {self.cagr:.2%}",
            f"  Sharpe:       {self.sharpe:.3f}",
            f"  Sortino:      {self.sortino:.3f}",
            f"  Max drawdown: {self.max_drawdown:.2%}",
            f"  Win rate:     {self.win_rate:.2%}",
            f"  Num trades:   {self.num_trades}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PortfolioResult(id={self.portfolio_id}, "
            f"symbols={self.symbols}, "
            f"return={self.total_return:.2%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioBacktester
# ─────────────────────────────────────────────────────────────────────────────


class PortfolioBacktester:
    """
    Multi-asset portfolio backtester.

    Usage
    -----
    ::

        from phinance.backtest.portfolio import PortfolioBacktester, PortfolioConfig

        backtester = PortfolioBacktester(
            ohlcv_dict={"SPY": spy_df, "QQQ": qqq_df},
            signals={"SPY": spy_signal, "QQQ": qqq_signal},
            config=PortfolioConfig(initial_capital=200_000),
        )
        result = backtester.run()
        print(result.summary())
    """

    def __init__(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.Series],
        config: Optional[PortfolioConfig] = None,
    ) -> None:
        self.ohlcv_dict = ohlcv_dict
        self.signals    = signals
        self.config     = config or PortfolioConfig()
        self.symbols    = list(ohlcv_dict.keys())

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> PortfolioResult:
        """Run the portfolio backtest and return a PortfolioResult."""
        t0 = time.perf_counter()

        # Align all series to common length (shortest)
        min_len = min(len(df) for df in self.ohlcv_dict.values())
        closes_dict = {
            sym: self.ohlcv_dict[sym]["close"].values[:min_len].astype(float)
            for sym in self.symbols
        }
        signals_dict = {
            sym: self.signals[sym].fillna(0.0).values[:min_len].astype(float)
            for sym in self.symbols
        }

        # Compute capital allocations
        allocs = self._compute_allocations(closes_dict, signals_dict)

        # Simulate each asset
        asset_equities: Dict[str, np.ndarray] = {}
        asset_results:  Dict[str, AssetResult] = {}

        for sym in self.symbols:
            eq, ar = self._simulate_asset(
                symbol=sym,
                closes=closes_dict[sym],
                signal=signals_dict[sym],
                capital=self.config.initial_capital * allocs[sym],
                allocation=allocs[sym],
            )
            asset_equities[sym] = eq
            asset_results[sym]  = ar

        # Aggregate portfolio equity
        portfolio_equity = self._aggregate_equity(asset_equities, min_len)
        port_metrics     = self._portfolio_metrics(portfolio_equity)

        # Correlation matrix of daily returns
        corr = self._correlation_matrix(asset_equities)

        elapsed = (time.perf_counter() - t0) * 1000.0
        total_trades = sum(ar.num_trades for ar in asset_results.values())
        all_wrs = [ar.win_rate for ar in asset_results.values() if ar.num_trades > 0]
        mean_wr = float(np.mean(all_wrs)) if all_wrs else 0.0

        return PortfolioResult(
            symbols=self.symbols,
            initial_capital=self.config.initial_capital,
            final_capital=float(portfolio_equity[-1]) if len(portfolio_equity) > 0 else self.config.initial_capital,
            total_return=port_metrics["total_return"],
            cagr=port_metrics["cagr"],
            sharpe=port_metrics["sharpe"],
            sortino=port_metrics["sortino"],
            max_drawdown=port_metrics["max_drawdown"],
            win_rate=mean_wr,
            num_trades=total_trades,
            portfolio_equity=portfolio_equity,
            asset_results=asset_results,
            correlation_matrix=corr,
            elapsed_ms=elapsed,
            config=self.config.to_dict(),
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _compute_allocations(
        self,
        closes_dict: Dict[str, np.ndarray],
        signals_dict: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Return per-asset capital fractions that sum to 1.0."""
        method = self.config.allocation
        n = len(self.symbols)

        if method == AllocationMethod.EQUAL or n == 0:
            return {s: 1.0 / n for s in self.symbols}

        if method == AllocationMethod.FIXED:
            fw = self.config.fixed_weights
            total = sum(fw.get(s, 0.0) for s in self.symbols)
            if total <= 0:
                return {s: 1.0 / n for s in self.symbols}
            return {s: fw.get(s, 0.0) / total for s in self.symbols}

        if method == AllocationMethod.RISK_PARITY:
            vols = {}
            for sym in self.symbols:
                c = closes_dict[sym]
                if len(c) > 1:
                    ret = np.diff(c) / (c[:-1] + 1e-9)
                    vols[sym] = float(ret.std()) or 1e-9
                else:
                    vols[sym] = 1e-9
            inv_vols = {s: 1.0 / v for s, v in vols.items()}
            total = sum(inv_vols.values())
            return {s: iv / total for s, iv in inv_vols.items()}

        # Fallback to equal
        return {s: 1.0 / n for s in self.symbols}

    def _simulate_asset(
        self,
        symbol: str,
        closes: np.ndarray,
        signal: np.ndarray,
        capital: float,
        allocation: float,
    ) -> tuple[np.ndarray, AssetResult]:
        """Simulate one asset and return (equity_curve, AssetResult)."""
        cfg       = self.config
        threshold = cfg.signal_threshold
        n         = len(closes)

        equity   = np.empty(n)
        equity[0] = capital
        position  = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        num_trades  = 0
        wins        = 0

        for i in range(1, n):
            sig = float(signal[i - 1])

            # Entry / exit logic
            new_pos = 0
            if sig >= threshold:
                new_pos = 1
            elif cfg.allow_short and sig <= -threshold:
                new_pos = -1

            # Trade on position change
            if new_pos != position:
                if position != 0:
                    # Close position
                    ret    = (closes[i] - entry_price) / entry_price * position
                    pnl    = equity[i - 1] * ret * cfg.position_size
                    equity[i] = equity[i - 1] + pnl - abs(equity[i - 1] * cfg.transaction_cost)
                    num_trades += 1
                    if pnl > 0:
                        wins += 1
                else:
                    equity[i] = equity[i - 1]

                if new_pos != 0:
                    entry_price = closes[i]
                    equity[i] = equity.get(i, equity[i - 1]) if hasattr(equity, "get") else equity[i]
                position = new_pos
            else:
                # Hold: mark to market
                if position != 0:
                    ret = (closes[i] - closes[i - 1]) / closes[i - 1] * position
                    equity[i] = equity[i - 1] * (1 + ret * cfg.position_size)
                else:
                    equity[i] = equity[i - 1]

        # Final close if still in position
        if position != 0 and n > 1:
            ret = (closes[-1] - entry_price) / entry_price * position
            pnl = equity[-1] * ret * cfg.position_size
            equity[-1] = equity[-1] + pnl - abs(equity[-1] * cfg.transaction_cost)
            num_trades += 1
            if pnl > 0:
                wins += 1

        # Clamp to >= 0
        equity = np.maximum(equity, 0.0)

        # Metrics
        ret_series = np.diff(equity) / (equity[:-1] + 1e-9)
        tr  = float((equity[-1] - equity[0]) / (equity[0] + 1e-9)) if n > 1 else 0.0
        sh  = float(ret_series.mean() / (ret_series.std() + 1e-9) * np.sqrt(252)) if len(ret_series) > 0 else 0.0
        sort_neg = ret_series[ret_series < 0]
        sort_std = float(sort_neg.std()) if len(sort_neg) > 1 else 1e-9
        so  = float(ret_series.mean() / (sort_std + 1e-9) * np.sqrt(252))
        peak = np.maximum.accumulate(equity)
        dd   = float(((peak - equity) / (peak + 1e-9)).max()) if n > 1 else 0.0
        wr   = float(wins / num_trades) if num_trades > 0 else 0.0

        ar = AssetResult(
            symbol=symbol,
            allocation=allocation,
            total_return=tr,
            sharpe=sh,
            max_drawdown=dd,
            win_rate=wr,
            num_trades=num_trades,
            equity_curve=equity,
        )
        return equity, ar

    def _aggregate_equity(
        self, asset_equities: Dict[str, np.ndarray], length: int
    ) -> np.ndarray:
        """Sum per-asset equity curves."""
        portfolio = np.zeros(length)
        for eq in asset_equities.values():
            portfolio += eq[:length]
        return portfolio

    def _portfolio_metrics(self, equity: np.ndarray) -> Dict[str, float]:
        """Compute portfolio-level metrics from the aggregate equity curve."""
        if len(equity) < 2:
            return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

        ret_series = np.diff(equity) / (equity[:-1] + 1e-9)
        tr  = float((equity[-1] - equity[0]) / (equity[0] + 1e-9))
        _cagr = float(((equity[-1] / equity[0]) ** (252.0 / len(equity)) - 1)) if equity[0] > 0 else 0.0
        std = float(ret_series.std()) or 1e-9
        sh  = float(ret_series.mean() / std * np.sqrt(252))
        neg = ret_series[ret_series < 0]
        neg_std = float(neg.std()) if len(neg) > 1 else 1e-9
        so  = float(ret_series.mean() / (neg_std + 1e-9) * np.sqrt(252))
        peak = np.maximum.accumulate(equity)
        dd   = float(((peak - equity) / (peak + 1e-9)).max())

        return {
            "total_return": tr,
            "cagr":         _cagr,
            "sharpe":       sh,
            "sortino":      so,
            "max_drawdown": dd,
        }

    def _correlation_matrix(self, asset_equities: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute correlation matrix of daily returns across assets."""
        if len(asset_equities) < 2:
            return np.array([[1.0]])
        returns = []
        for sym in self.symbols:
            eq = asset_equities.get(sym, np.array([]))
            if len(eq) > 1:
                returns.append(np.diff(eq) / (eq[:-1] + 1e-9))
        if not returns:
            return np.eye(len(self.symbols))
        min_len = min(len(r) for r in returns)
        mat = np.stack([r[:min_len] for r in returns])
        try:
            return np.corrcoef(mat)
        except Exception:  # noqa: BLE001
            return np.eye(len(self.symbols))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def run_portfolio_backtest(
    ohlcv_dict: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.Series],
    initial_capital: float = 100_000.0,
    allocation: str = "equal",
    transaction_cost: float = 0.001,
    allow_short: bool = False,
    fixed_weights: Optional[Dict[str, float]] = None,
) -> PortfolioResult:
    """
    One-shot portfolio backtest.

    Parameters
    ----------
    ohlcv_dict       : dict  — {symbol: OHLCV DataFrame}
    signals          : dict  — {symbol: signal Series in [-1, 1]}
    initial_capital  : float
    allocation       : str   — "equal" | "risk_parity" | "fixed"
    transaction_cost : float
    allow_short      : bool
    fixed_weights    : dict or None — required when allocation="fixed"

    Returns
    -------
    PortfolioResult
    """
    method_map = {
        "equal":       AllocationMethod.EQUAL,
        "risk_parity": AllocationMethod.RISK_PARITY,
        "fixed":       AllocationMethod.FIXED,
    }
    cfg = PortfolioConfig(
        initial_capital=initial_capital,
        allocation=method_map.get(allocation, AllocationMethod.EQUAL),
        fixed_weights=fixed_weights or {},
        transaction_cost=transaction_cost,
        allow_short=allow_short,
    )
    return PortfolioBacktester(ohlcv_dict=ohlcv_dict, signals=signals, config=cfg).run()
