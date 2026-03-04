"""
phinance.live.paper_engine
===========================

Pure in-process paper trading broker.

PaperBroker simulates order execution locally — no external API calls,
no network required.  It uses the last known price from ``update_price()``
to fill market orders immediately at that price.

This is the default broker for testing and dry-runs.

Features
--------
* Instant market-order fills at current price (no slippage by default)
* Configurable slippage model (fraction of price, default 0.0005)
* Commission model (per-share, default $0.01)
* Tracks cash, positions, equity, and a full trade history
* Thread-safe via internal ``threading.Lock``

Usage
-----
    from phinance.live.paper_engine import PaperBroker
    from phinance.live.broker_base import OrderSide

    broker = PaperBroker(initial_capital=100_000)
    broker.connect()
    broker.update_price("SPY", 450.0)
    order = broker.submit_order("SPY", 10, OrderSide.BUY)
    print(broker.get_account())   # {'equity': 100_000, 'cash': 95_500, ...}
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from phinance.live.broker_base import BrokerAdapter, OrderSide, OrderType, OrderStatus
from phinance.live.order_models import Order, Fill, Position
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


class PaperBroker(BrokerAdapter):
    """In-process paper trading broker.

    Parameters
    ----------
    initial_capital : float — starting cash (default 100 000)
    slippage        : float — fraction of price added/subtracted on fill
                              (default 0.0005 = 0.05 %)
    commission      : float — per-share commission (default 0.01)
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage: float = 0.0005,
        commission: float = 0.01,
    ) -> None:
        self._capital   = float(initial_capital)
        self._cash      = float(initial_capital)
        self._slippage  = float(slippage)
        self._commission = float(commission)

        self._positions: Dict[str, Position] = {}
        self._orders:    Dict[str, Order]    = {}
        self._fills:     List[Fill]          = []
        self._prices:    Dict[str, float]    = {}

        self._lock = threading.Lock()
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        logger.info("PaperBroker connected (capital=%.2f)", self._capital)

    def disconnect(self) -> None:
        self._connected = False

    # ── Price update ──────────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current market price for a symbol."""
        with self._lock:
            self._prices[symbol] = float(price)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Bulk price update."""
        with self._lock:
            for sym, px in prices.items():
                self._prices[sym] = float(px)

    # ── Account / market data ─────────────────────────────────────────────────

    def get_account(self) -> Dict:
        with self._lock:
            equity = self._cash + sum(
                pos.qty * self._prices.get(sym, pos.avg_entry)
                for sym, pos in self._positions.items()
            )
            return {
                "equity":        round(equity, 2),
                "cash":          round(self._cash, 2),
                "initial_capital": round(self._capital, 2),
                "total_pnl":     round(equity - self._capital, 2),
                "num_positions": len([p for p in self._positions.values() if p.qty != 0]),
            }

    def get_positions(self) -> List[Position]:
        with self._lock:
            positions = []
            for sym, pos in self._positions.items():
                if pos.qty == 0:
                    continue
                cur_price = self._prices.get(sym, pos.avg_entry)
                pos.current_price  = cur_price
                pos.market_value   = pos.qty * cur_price
                pos.unrealised_pl  = (cur_price - pos.avg_entry) * pos.qty
                positions.append(pos)
            return positions

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        with self._lock:
            price = self._prices.get(symbol)
        if price is None:
            return None
        return pd.Series({
            "open": price, "high": price, "low": price,
            "close": price, "volume": 0,
        })

    # ── Order management ──────────────────────────────────────────────────────

    def submit_order(
        self,
        symbol:          str,
        qty:             float,
        side:            OrderSide,
        order_type:      OrderType = OrderType.MARKET,
        limit_price:     Optional[float] = None,
        stop_price:      Optional[float] = None,
        time_in_force:   str = "day",
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Submit an order and immediately fill market orders at current price."""
        order_id = str(uuid.uuid4())
        order = Order(
            order_id        = order_id,
            symbol          = symbol,
            side            = side.value if isinstance(side, OrderSide) else str(side),
            qty             = float(qty),
            order_type      = order_type.value if isinstance(order_type, OrderType) else str(order_type),
            status          = OrderStatus.SUBMITTED.value,
            limit_price     = limit_price,
            stop_price      = stop_price,
            time_in_force   = time_in_force,
            client_order_id = client_order_id,
            submitted_at    = datetime.utcnow(),
        )

        with self._lock:
            self._orders[order_id] = order

        if order_type in (OrderType.MARKET, "market"):
            self._fill_market_order(order)

        return order

    def _fill_market_order(self, order: Order) -> None:
        """Execute a market order immediately at current price ± slippage."""
        with self._lock:
            symbol = order.symbol
            price  = self._prices.get(symbol)
            if price is None:
                order.status = OrderStatus.REJECTED.value
                logger.warning("No price available for %s — order rejected", symbol)
                return

            # Apply slippage
            if order.side == "buy":
                fill_price = price * (1 + self._slippage)
            else:
                fill_price = price * (1 - self._slippage)

            qty        = order.qty
            commission = qty * self._commission
            cost       = (fill_price * qty) + commission

            if order.side == "buy":
                if self._cash < cost:
                    # Partial fill if insufficient cash
                    max_qty = int(self._cash / (fill_price + self._commission))
                    if max_qty <= 0:
                        order.status = OrderStatus.REJECTED.value
                        logger.warning("Insufficient cash for %s order", symbol)
                        return
                    qty      = float(max_qty)
                    cost     = fill_price * qty + qty * self._commission

                self._cash -= cost
                # Update position
                pos = self._positions.get(symbol, Position(symbol=symbol, qty=0, avg_entry=0.0))
                new_qty = pos.qty + qty
                if pos.qty == 0:
                    pos.avg_entry = fill_price
                else:
                    pos.avg_entry = (pos.avg_entry * pos.qty + fill_price * qty) / new_qty
                pos.qty = new_qty
                self._positions[symbol] = pos

            else:  # sell
                pos = self._positions.get(symbol)
                sell_qty = min(qty, pos.qty if pos else 0)
                if sell_qty <= 0:
                    order.status = OrderStatus.REJECTED.value
                    logger.warning("No position to sell for %s", symbol)
                    return
                proceeds      = fill_price * sell_qty - sell_qty * self._commission
                self._cash   += proceeds
                pos.qty      -= sell_qty
                if pos.qty <= 0:
                    pos.qty = 0

            # Record fill
            fill = Fill(
                order_id  = order.order_id,
                symbol    = symbol,
                side      = order.side,
                qty       = qty,
                price     = fill_price,
                filled_at = datetime.utcnow(),
                commission = qty * self._commission,
            )
            self._fills.append(fill)

            order.status          = OrderStatus.FILLED.value
            order.filled_qty      = qty
            order.filled_avg_price = fill_price
            order.filled_at        = datetime.utcnow()

            logger.info(
                "PaperBroker FILL: %s %s %.0f@%.2f (commission %.2f)",
                symbol, order.side, qty, fill_price, qty * self._commission,
            )

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order and order.is_pending:
                order.status = OrderStatus.CANCELLED.value
                return True
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders[-limit:]

    @property
    def is_paper(self) -> bool:
        return True

    def get_fills(self) -> List[Fill]:
        """Return all fills (trade executions)."""
        return list(self._fills)

    def reset(self) -> None:
        """Reset the paper broker to its initial state."""
        with self._lock:
            self._cash       = self._capital
            self._positions  = {}
            self._orders     = {}
            self._fills      = []
