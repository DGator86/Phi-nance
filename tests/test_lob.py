"""Unit tests for LOB simulation components."""

from __future__ import annotations

from itertools import islice

import pandas as pd

from phi.lob.book import OrderBook
from phi.lob.data import LobEvent, iter_events
from phi.lob.engine import LobSimEngine
from phi.lob.order import OrderSide, OrderType, SimOrder
from phi.lob.strategy import LobStrategy
from phi.lob.synthetic import generate_synthetic_events


class _BuyOnceStrategy(LobStrategy):
    def __init__(self) -> None:
        self.done = False

    def on_event(self, event: LobEvent, book: OrderBook, portfolio):
        if not self.done and book.best_ask() is not None:
            self.done = True
            return [SimOrder(side=OrderSide.BUY, quantity=2.0, order_type=OrderType.MARKET)]
        return []


class _LimitCancelStrategy(LobStrategy):
    def __init__(self) -> None:
        self.stage = 0

    def on_event(self, event: LobEvent, book: OrderBook, portfolio):
        if self.stage == 0 and book.best_bid() is not None:
            self.stage = 1
            return [
                SimOrder(
                    side=OrderSide.BUY,
                    quantity=1.0,
                    order_type=OrderType.LIMIT,
                    price=book.best_bid() - 0.5,
                    order_id="rest1",
                )
            ]
        if self.stage == 1:
            self.stage = 2
            return [
                SimOrder(
                    side=OrderSide.BUY,
                    quantity=1.0,
                    order_type=OrderType.CANCEL,
                    price=book.best_bid() - 0.5,
                    order_id="rest1",
                )
            ]
        return []


def _sample_events() -> list[LobEvent]:
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    return [
        LobEvent(timestamp=base, event_type="add", price=100.0, volume=5.0, side="buy"),
        LobEvent(timestamp=base + pd.Timedelta(seconds=1), event_type="add", price=101.0, volume=6.0, side="sell"),
        LobEvent(timestamp=base + pd.Timedelta(seconds=2), event_type="trade", price=101.0, volume=1.0, side="buy"),
    ]


def test_order_book_updates_and_depth() -> None:
    book = OrderBook()
    for event in _sample_events()[:2]:
        book.process_event(event)

    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0
    assert book.spread() == 1.0
    assert book.mid_price() == 100.5
    depth = book.market_depth(1)
    assert depth["bids"] == [(100.0, 5.0)]
    assert depth["asks"] == [(101.0, 6.0)]


def test_market_order_execution() -> None:
    engine = LobSimEngine(_sample_events(), _BuyOnceStrategy(), initial_capital=1000.0)
    result = engine.run()

    assert result.metrics["trades"] >= 1
    assert engine.portfolio.positions.get("SIM", 0.0) > 0


def test_limit_placement_and_cancellation() -> None:
    events = _sample_events()
    engine = LobSimEngine(events, _LimitCancelStrategy(), initial_capital=1000.0)
    engine.run()
    assert "rest1" not in engine.book.resting_orders


def test_synthetic_generator_reasonable_events() -> None:
    events = list(islice(generate_synthetic_events(seed=7), 10))
    assert len(events) == 10
    assert all(event.price > 0 for event in events)
    assert all(event.volume > 0 for event in events)


def test_iter_events_from_dataframe() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"],
            "event_type": ["add", "add"],
            "price": [100.0, 101.0],
            "volume": [2.0, 3.0],
            "side": ["buy", "sell"],
        }
    )
    events = list(iter_events(df))
    assert len(events) == 2
    assert events[0].event_type == "add"
