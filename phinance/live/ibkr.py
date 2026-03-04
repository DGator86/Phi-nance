"""
phinance.live.ibkr
===================

Interactive Brokers broker adapter (TWS / IB Gateway).

Uses the ``ib_insync`` library for a clean async/sync API over the IB socket.
Falls back to stub mode if ``ib_insync`` is not installed so other modules
can import without crashing.

Prerequisites
-------------
    pip install ib_insync>=0.9.86

And either TWS or IB Gateway must be running locally:
    Paper TWS:     port 7497, Client ID: 1
    Paper Gateway: port 4002, Client ID: 1
    Live TWS:      port 7496
    Live Gateway:  port 4001

Environment variables
---------------------
    IBKR_HOST        — IB Gateway host (default: 127.0.0.1)
    IBKR_PORT        — IB Gateway port (default: 7497  ← paper TWS)
    IBKR_CLIENT_ID   — Client ID (default: 1)
    IBKR_ACCOUNT     — Account number (default: first managed account)

References
----------
* https://ib-insync.readthedocs.io/
* https://interactivebrokers.github.io/tws-api/
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from phinance.live.broker_base import BrokerAdapter, OrderSide, OrderType, OrderStatus
from phinance.live.order_models import Order, Fill, Position
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import ib_insync as ibi
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False


class IBKRBroker(BrokerAdapter):
    """Interactive Brokers broker adapter via ib_insync.

    Parameters
    ----------
    host      : str — TWS/Gateway host (default ``IBKR_HOST`` or ``127.0.0.1``)
    port      : int — TWS/Gateway port (default ``IBKR_PORT`` or 7497)
    client_id : int — client identifier (default ``IBKR_CLIENT_ID`` or 1)
    account   : str — IB account number (default ``IBKR_ACCOUNT`` or auto)
    readonly  : bool — connect in read-only mode (default False)
    """

    def __init__(
        self,
        host:      Optional[str] = None,
        port:      Optional[int] = None,
        client_id: Optional[int] = None,
        account:   Optional[str] = None,
        readonly:  bool = False,
    ) -> None:
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IBKRBroker. "
                "Install with: pip install ib_insync"
            )
        self._host      = host      or os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port      = int(port  or os.environ.get("IBKR_PORT", 7497))
        self._client_id = int(client_id or os.environ.get("IBKR_CLIENT_ID", 1))
        self._account   = account   or os.environ.get("IBKR_ACCOUNT", "")
        self._readonly  = readonly

        self._ib: Optional["ibi.IB"] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._ib = ibi.IB()
        self._ib.connect(
            self._host, self._port,
            clientId=self._client_id,
            readonly=self._readonly,
        )
        managed = self._ib.managedAccounts()
        if not self._account and managed:
            self._account = managed[0]
        logger.info(
            "IBKRBroker connected: host=%s port=%d account=%s",
            self._host, self._port, self._account,
        )

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None

    # ── Account / market data ─────────────────────────────────────────────────

    def get_account(self) -> Dict:
        if not self._ib:
            raise RuntimeError("Not connected")
        summary = self._ib.accountSummary(self._account)
        result: Dict = {}
        for entry in summary:
            result[entry.tag] = entry.value
        return result

    def get_positions(self) -> List[Position]:
        if not self._ib:
            raise RuntimeError("Not connected")
        positions = []
        for pos in self._ib.positions(self._account):
            positions.append(Position(
                symbol    = pos.contract.symbol,
                qty       = float(pos.position),
                avg_entry = float(pos.avgCost),
            ))
        return positions

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        if not self._ib:
            return None
        try:
            contract = ibi.Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None
            bar = bars[-1]
            return pd.Series({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })
        except Exception as exc:
            logger.warning("IBKRBroker.get_latest_bar(%s) failed: %s", symbol, exc)
            return None

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
        if not self._ib:
            raise RuntimeError("Not connected")

        contract   = ibi.Stock(symbol, "SMART", "USD")
        action     = "BUY" if side in (OrderSide.BUY, "buy") else "SELL"
        tif        = "DAY" if time_in_force == "day" else "GTC"

        if order_type in (OrderType.MARKET, "market"):
            ib_order = ibi.MarketOrder(action=action, totalQuantity=qty, tif=tif)
        elif order_type in (OrderType.LIMIT, "limit"):
            ib_order = ibi.LimitOrder(action=action, totalQuantity=qty,
                                      lmtPrice=limit_price, tif=tif)
        else:
            ib_order = ibi.StopOrder(action=action, totalQuantity=qty,
                                     stopPrice=stop_price, tif=tif)

        trade = self._ib.placeOrder(contract, ib_order)
        self._ib.sleep(0.1)   # allow status update

        status = "submitted"
        if trade.orderStatus.status in ("Filled", "PreSubmitted"):
            status = "filled"

        return Order(
            order_id     = str(trade.order.orderId),
            symbol       = symbol,
            side         = action.lower(),
            qty          = float(qty),
            order_type   = str(order_type),
            status       = status,
            limit_price  = limit_price,
            stop_price   = stop_price,
            time_in_force = time_in_force,
            submitted_at  = datetime.utcnow(),
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self._ib:
            return False
        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
        except Exception:
            pass
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        if not self._ib:
            return None
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                status = "submitted"
                if trade.orderStatus.status == "Filled":
                    status = "filled"
                elif trade.orderStatus.status == "Cancelled":
                    status = "cancelled"
                return Order(
                    order_id  = order_id,
                    symbol    = trade.contract.symbol,
                    side      = trade.order.action.lower(),
                    qty       = float(trade.order.totalQuantity),
                    status    = status,
                )
        return None

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        if not self._ib:
            return []
        result = []
        for trade in self._ib.trades()[:limit]:
            o_status = "submitted"
            if trade.orderStatus.status == "Filled":
                o_status = "filled"
            if status and o_status != status:
                continue
            result.append(Order(
                order_id = str(trade.order.orderId),
                symbol   = trade.contract.symbol,
                side     = trade.order.action.lower(),
                qty      = float(trade.order.totalQuantity),
                status   = o_status,
            ))
        return result

    @property
    def is_paper(self) -> bool:
        return self._port in (7497, 4002)   # paper ports
