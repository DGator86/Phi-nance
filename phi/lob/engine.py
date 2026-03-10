"""Event-driven limit order book simulation engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phi.lob.portfolio_adapter import Order as PortfolioOrder
from phi.lob.portfolio_adapter import Portfolio
from phi.lob.book import OrderBook
from phi.lob.data import LobEvent, ensure_event_iterator
from phi.lob.order import Fill, OrderSide, OrderType, SimOrder
from phi.lob.strategy import LobStrategy


@dataclass
class LobSimResult:
    """Result payload returned by :class:`LobSimEngine`."""

    metrics: dict[str, float]
    fills: list[Fill]
    equity_curve: pd.DataFrame


class LobSimEngine:
    """Replay LOB events, invoke strategy, and maintain portfolio state."""

    def __init__(
        self,
        events: Iterable[LobEvent] | pd.DataFrame,
        strategy: LobStrategy,
        *,
        symbol: str = "SIM",
        initial_capital: float = 100_000.0,
    ) -> None:
        self.events = ensure_event_iterator(events)
        self.strategy = strategy
        self.symbol = symbol
        self.book = OrderBook()
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.fills: list[Fill] = []
        self.current_time = pd.Timestamp.utcnow()

    def _record_fill(self, fill: Fill) -> None:
        shares = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        self.portfolio.execute_order(
            PortfolioOrder(
                symbol=self.symbol,
                shares=shares,
                price=fill.price,
                timestamp=fill.timestamp,
            )
        )
        self.fills.append(fill)

    def _execute_market(self, order: SimOrder) -> list[Fill]:
        remaining = max(0.0, float(order.quantity))
        fills: list[Fill] = []
        if remaining <= 0:
            return fills

        if order.side == OrderSide.BUY:
            levels = sorted(self.book.asks.items(), key=lambda item: item[0])
            consume_side = "sell"
        else:
            levels = sorted(self.book.bids.items(), key=lambda item: item[0], reverse=True)
            consume_side = "buy"

        for price, available in levels:
            if remaining <= 0:
                break
            qty = min(remaining, available)
            removed = self.book.reduce_level(consume_side, price, qty)
            if removed <= 0:
                continue
            fill = Fill(
                side=order.side,
                quantity=removed,
                price=price,
                order_id=order.order_id,
                symbol=order.symbol,
                timestamp=order.timestamp or self.current_time,
            )
            fills.append(fill)
            remaining -= removed
        return fills

    def _execute_limit(self, order: SimOrder) -> list[Fill]:
        if order.price is None:
            return []
        crossing = (
            order.side == OrderSide.BUY and self.book.best_ask() is not None and order.price >= float(self.book.best_ask())
        ) or (
            order.side == OrderSide.SELL and self.book.best_bid() is not None and order.price <= float(self.book.best_bid())
        )

        fills: list[Fill] = []
        if crossing:
            market_like = SimOrder(
                side=order.side,
                quantity=order.quantity,
                order_type=OrderType.MARKET,
                order_id=order.order_id,
                symbol=order.symbol,
                timestamp=order.timestamp,
            )
            fills.extend(self._execute_market(market_like))
            filled_qty = sum(fill.quantity for fill in fills)
            remaining = max(0.0, order.quantity - filled_qty)
            if remaining > 0:
                self.book.add_level(order.side, float(order.price), remaining)
            return fills

        self.book.add_level(order.side, float(order.price), order.quantity)
        if order.order_id:
            self.book.resting_orders[order.order_id] = (order.side, float(order.price), order.quantity)
        return fills

    def _execute_cancel(self, order: SimOrder) -> None:
        if order.order_id and order.order_id in self.book.resting_orders:
            side, price, _quantity = self.book.resting_orders.pop(order.order_id)
            self.book.reduce_level(side, price, order.quantity)
            return
        if order.price is not None:
            self.book.reduce_level(order.side, float(order.price), order.quantity)

    def execute_strategy_orders(self, orders: list[SimOrder], timestamp: pd.Timestamp) -> None:
        for order in orders:
            order.timestamp = order.timestamp or timestamp
            order.symbol = order.symbol or self.symbol
            if order.order_type == OrderType.MARKET:
                for fill in self._execute_market(order):
                    self._record_fill(fill)
            elif order.order_type == OrderType.LIMIT:
                for fill in self._execute_limit(order):
                    self._record_fill(fill)
            elif order.order_type == OrderType.CANCEL:
                self._execute_cancel(order)

    def _compute_metrics(self) -> dict[str, float]:
        if not self.portfolio.equity_curve:
            return {"pnl": 0.0, "return": 0.0, "sharpe": 0.0, "trades": 0.0}

        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
        start = float(equity_df["equity"].iloc[0])
        end = float(equity_df["equity"].iloc[-1])
        rets = equity_df["equity"].pct_change().dropna()
        sharpe = 0.0
        if not rets.empty and rets.std() > 1e-12:
            sharpe = float(np.sqrt(252.0) * rets.mean() / rets.std())
        return {
            "pnl": end - start,
            "return": (end / start - 1.0) if start else 0.0,
            "sharpe": sharpe,
            "trades": float(len(self.fills)),
        }

    def run(self, max_events: int | None = None) -> LobSimResult:
        """Run the simulation loop until events exhaust or max_events reached."""
        self.strategy.on_start(self.book, self.portfolio)
        count = 0
        for event in self.events:
            self.current_time = pd.Timestamp(event.timestamp)
            self.book.process_event(event)
            mid = self.book.mid_price() or float(event.price)
            self.portfolio.update_prices({self.symbol: mid})
            strategy_orders = self.strategy.on_event(event, self.book, self.portfolio)
            self.execute_strategy_orders(strategy_orders or [], self.current_time)
            self.portfolio.update_prices({self.symbol: self.book.mid_price() or float(event.price)})
            self.portfolio.record_equity(self.current_time)
            count += 1
            if max_events is not None and count >= max_events:
                break

        self.strategy.on_finish(self.book, self.portfolio)
        equity_curve = pd.DataFrame(self.portfolio.equity_curve, columns=["timestamp", "equity"])
        return LobSimResult(metrics=self._compute_metrics(), fills=self.fills, equity_curve=equity_curve)
