"""Options-aware portfolio accounting."""

from __future__ import annotations

from datetime import date
from typing import Dict

from .position import OptionPosition


class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = float(initial_cash)
        self.equity_positions: Dict[str, dict] = {}
        self.option_positions: list[OptionPosition] = []
        self.equity_history: list[tuple[date, float]] = []
        self.trade_log: list[dict] = []

    def open_option(self, position: OptionPosition) -> None:
        """Open a new options position."""
        self.option_positions.append(position)
        self.cash -= position.quantity * position.entry_price * position.multiplier

    def close_option(self, position: OptionPosition, exit_price: float, exit_date: date) -> None:
        """Close an existing options position."""
        position.close(exit_price, exit_date)
        self.cash += position.quantity * exit_price * position.multiplier
        self.trade_log.append(
            {
                "symbol": position.symbol,
                "option_type": position.option_type,
                "strike": position.strike,
                "expiration": position.expiration,
                "entry_date": position.entry_date,
                "exit_date": exit_date,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "quantity": position.quantity,
                "pnl": (exit_price - position.entry_price) * position.quantity * position.multiplier,
            }
        )

    def snapshot(self, current_date: date, underlying_price: float, option_prices: Dict[str, float]) -> float:
        """Record current total equity (cash + options MTM)."""
        _ = underlying_price
        options_value = self.mark_to_market_options(current_date, option_prices)
        total_equity = self.cash + options_value
        self.equity_history.append((current_date, total_equity))
        return total_equity

    def mark_to_market_options(self, current_date: date, prices: Dict[str, float]) -> float:
        """Mark all open options positions to market and return total option value."""
        total = 0.0
        for pos in self.option_positions:
            if pos.exit_date is not None or pos.is_expired(current_date):
                continue
            key = f"{pos.symbol}_{pos.strike}_{pos.expiration}_{pos.option_type}"
            total += pos.mark_to_market(prices.get(key, 0.0))
        return total

    def settle_expirations(self, current_date: date, underlying_prices: Dict[str, float]) -> None:
        """Settle all expired options using intrinsic value cash settlement."""
        for pos in self.option_positions:
            if pos.exit_date is not None or not pos.is_expired(current_date):
                continue

            underlying_price = float(underlying_prices.get(pos.symbol, 0.0))
            if pos.option_type == "CALL" and underlying_price > pos.strike:
                intrinsic = (underlying_price - pos.strike) * abs(pos.quantity) * pos.multiplier
            elif pos.option_type == "PUT" and underlying_price < pos.strike:
                intrinsic = (pos.strike - underlying_price) * abs(pos.quantity) * pos.multiplier
            else:
                intrinsic = 0.0

            self.cash += intrinsic if pos.is_long else -intrinsic
            per_contract = intrinsic / (abs(pos.quantity) * pos.multiplier) if abs(pos.quantity) else 0.0
            pos.close(exit_price=per_contract, exit_date=current_date)
