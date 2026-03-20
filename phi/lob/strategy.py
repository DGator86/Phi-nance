"""Strategy interfaces and reference implementations for LOB simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phi.lob.portfolio_adapter import Portfolio
from phi.lob.book import OrderBook
from phi.lob.data import LobEvent
from phi.lob.order import OrderSide, OrderType, SimOrder


class LobStrategy(ABC):
    """Base class for event-driven LOB strategies."""

    def on_start(self, book: OrderBook, portfolio: Portfolio) -> None:
        """Hook called before simulation begins."""

    @abstractmethod
    def on_event(self, event: LobEvent, book: OrderBook, portfolio: Portfolio) -> list[SimOrder]:
        """Process one event and return strategy orders."""

    def on_finish(self, book: OrderBook, portfolio: Portfolio) -> None:
        """Hook called after simulation ends."""


class MarketMakingStrategy(LobStrategy):
    """Simple strategy posting buy/sell limits around mid-price."""

    def __init__(self, spread_bps: float = 5.0, size: float = 1.0) -> None:
        self.spread_bps = float(spread_bps)
        self.size = float(size)

    def on_event(self, event: LobEvent, book: OrderBook, portfolio: Portfolio) -> list[SimOrder]:
        mid = book.mid_price()
        if mid is None:
            return []
        offset = mid * self.spread_bps / 10_000.0
        return [
            SimOrder(side=OrderSide.BUY, quantity=self.size, order_type=OrderType.LIMIT, price=mid - offset),
            SimOrder(side=OrderSide.SELL, quantity=self.size, order_type=OrderType.LIMIT, price=mid + offset),
        ]


class ImbalanceStrategy(LobStrategy):
    """Trade with market orders when top-of-book imbalance is significant."""

    def __init__(self, threshold: float = 0.25, size: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.size = float(size)

    def on_event(self, event: LobEvent, book: OrderBook, portfolio: Portfolio) -> list[SimOrder]:
        bid = book.best_bid()
        ask = book.best_ask()
        if bid is None or ask is None:
            return []
        bid_v = book.bids.get(bid, 0.0)
        ask_v = book.asks.get(ask, 0.0)
        denom = bid_v + ask_v
        if denom <= 0:
            return []
        imbalance = (bid_v - ask_v) / denom
        if imbalance > self.threshold:
            return [SimOrder(side=OrderSide.BUY, quantity=self.size, order_type=OrderType.MARKET)]
        if imbalance < -self.threshold:
            return [SimOrder(side=OrderSide.SELL, quantity=self.size, order_type=OrderType.MARKET)]
        return []
