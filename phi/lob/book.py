"""In-memory limit order book data structure."""

from __future__ import annotations

from dataclasses import dataclass, field

from phi.lob.data import LobEvent
from phi.lob.order import OrderSide


@dataclass
class OrderBook:
    """Price-level order book with optional tracked resting strategy orders."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    resting_orders: dict[str, tuple[OrderSide, float, float]] = field(default_factory=dict)

    def _book_for_side(self, side: str | OrderSide) -> dict[float, float]:
        side_value = side.value if isinstance(side, OrderSide) else str(side).lower()
        return self.bids if side_value == "buy" else self.asks

    def _clean_level(self, side: str | OrderSide, price: float) -> None:
        book = self._book_for_side(side)
        if book.get(price, 0.0) <= 1e-12:
            book.pop(price, None)

    def add_level(self, side: str | OrderSide, price: float, volume: float) -> None:
        book = self._book_for_side(side)
        book[price] = book.get(price, 0.0) + max(0.0, float(volume))
        self._clean_level(side, price)

    def reduce_level(self, side: str | OrderSide, price: float, volume: float) -> float:
        book = self._book_for_side(side)
        available = book.get(price, 0.0)
        removed = min(available, max(0.0, float(volume)))
        book[price] = available - removed
        self._clean_level(side, price)
        return removed

    def process_event(self, event: LobEvent) -> None:
        """Apply market data event to the book."""
        kind = event.event_type.lower()
        if kind in {"add", "bid", "ask", "quote"}:
            self.add_level(event.side, event.price, event.volume)
            return
        if kind in {"cancel", "delete"}:
            self.reduce_level(event.side, event.price, event.volume)
            return
        if kind == "trade":
            opposite_side = "sell" if event.side == "buy" else "buy"
            self.reduce_level(opposite_side, event.price, event.volume)

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def spread(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def mid_price(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None and ask is None:
            return None
        if bid is None:
            return ask
        if ask is None:
            return bid
        return (bid + ask) / 2.0

    def market_depth(self, levels: int = 5) -> dict[str, list[tuple[float, float]]]:
        """Return top-of-book depth levels."""
        bids = sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[:levels]
        asks = sorted(self.asks.items(), key=lambda item: item[0])[:levels]
        return {"bids": bids, "asks": asks}
