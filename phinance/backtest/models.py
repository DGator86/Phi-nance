"""
phinance.backtest.models
=========================

Data classes for backtest results, trades, and positions.

Classes
-------
  Trade         — A single buy→sell round-trip
  Position      — Open position state at a point in time
  BacktestResult — Complete output of a backtest run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Trade:
    """A single completed round-trip trade.

    Attributes
    ----------
    entry_date   : Entry bar timestamp
    exit_date    : Exit bar timestamp
    symbol       : Ticker symbol
    entry_price  : Fill price at entry
    exit_price   : Fill price at exit
    quantity     : Number of shares / contracts
    pnl          : Absolute P&L (exit_price - entry_price) * quantity
    pnl_pct      : Fractional P&L relative to entry cost
    hold_bars    : Number of bars held
    direction    : ``"long"`` or ``"short"``
    regime       : Market regime label at entry (string)
    """

    entry_date:  Any
    exit_date:   Any
    symbol:      str
    entry_price: float
    exit_price:  float
    quantity:    int
    pnl:         float
    pnl_pct:     float
    hold_bars:   int
    direction:   str = "long"
    regime:      str = "UNKNOWN"

    @property
    def win(self) -> bool:
        """True if this trade is profitable."""
        return self.pnl > 0


@dataclass
class Position:
    """Current open position snapshot.

    Attributes
    ----------
    symbol       : Ticker
    quantity     : Shares held (negative = short)
    entry_price  : Average entry price
    entry_date   : Entry bar timestamp
    current_price : Latest market price
    unrealised_pnl : Current unrealised P&L
    """

    symbol:         str
    quantity:       int
    entry_price:    float
    entry_date:     Any
    current_price:  float = 0.0
    unrealised_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest output.

    Attributes
    ----------
    symbol          : Primary symbol
    total_return    : Fractional total return (0.12 = 12 %)
    cagr            : Compound annual growth rate
    max_drawdown    : Maximum peak-to-trough drawdown (positive fraction)
    sharpe          : Annualised Sharpe ratio
    sortino         : Annualised Sortino ratio (0.0 if not computed)
    win_rate        : Fraction of winning trades
    total_trades    : Total closed trades
    portfolio_value : List of portfolio NAV over time
    net_pl          : Net dollar P&L
    trades          : List of Trade objects
    prediction_log  : Bar-by-bar signal log (for Phibot)
    metadata        : Free-form dict for extra context
    """

    symbol:          str = ""
    total_return:    float = 0.0
    cagr:            float = 0.0
    max_drawdown:    float = 0.0
    sharpe:          float = 0.0
    sortino:         float = 0.0
    win_rate:        float = 0.0
    total_trades:    int = 0
    portfolio_value: List[float] = field(default_factory=list)
    net_pl:          float = 0.0
    trades:          List[Trade] = field(default_factory=list)
    prediction_log:  List[Dict[str, Any]] = field(default_factory=list)
    metadata:        Dict[str, Any] = field(default_factory=dict)

    # Compat alias used by display code that reads "prediction_log"
    @property
    def _prediction_log(self) -> List[Dict[str, Any]]:
        return self.prediction_log

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "symbol":        self.symbol,
            "total_return":  self.total_return,
            "cagr":          self.cagr,
            "max_drawdown":  self.max_drawdown,
            "sharpe":        self.sharpe,
            "sortino":       self.sortino,
            "win_rate":      self.win_rate,
            "total_trades":  self.total_trades,
            "net_pl":        self.net_pl,
            "portfolio_value": self.portfolio_value,
            **self.metadata,
        }
