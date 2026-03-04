"""
tests.unit.test_live_trading_extended
=======================================

Extended unit tests for the live trading framework:
  • PaperBroker  — order submission, fills, slippage, commission, positions
  • BrokerAdapter abstract base (coverage of enums and protocol)
  • Order / Fill / Position dataclasses
  • LiveTradingLoop — seed_buffer, run_once, signal/decision pipeline

All tests are pure-unit: no network calls, no real broker connections.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

# Enums live in broker_base
from phinance.live.broker_base import OrderSide, OrderType, OrderStatus
from phinance.live.order_models import Order, Fill, Position
from phinance.live.paper_engine import PaperBroker
from phinance.live.trading_loop import LiveTradingLoop, LiveRunResult

# ── Fixtures ──────────────────────────────────────────────────────────────────

DF_60  = make_ohlcv(n=60)
DF_100 = make_ohlcv(n=100)


def _fresh_broker(capital: float = 100_000) -> PaperBroker:
    b = PaperBroker(initial_capital=capital)
    b.connect()
    return b


# ═══════════════════════════════════════════════════════════════════════════════
# OrderSide / OrderType / OrderStatus enums
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnums:

    def test_order_side_buy_value(self):
        assert OrderSide.BUY.value == "buy"

    def test_order_side_sell_value(self):
        assert OrderSide.SELL.value == "sell"

    def test_order_type_market_value(self):
        assert OrderType.MARKET.value == "market"

    def test_order_type_limit_value(self):
        assert OrderType.LIMIT.value == "limit"

    def test_order_status_pending(self):
        assert OrderStatus.PENDING.value == "pending"

    def test_order_status_filled(self):
        assert OrderStatus.FILLED.value == "filled"

    def test_order_status_cancelled(self):
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_order_status_submitted(self):
        assert OrderStatus.SUBMITTED.value == "submitted"


# ═══════════════════════════════════════════════════════════════════════════════
# Order dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderDataclass:

    def _make_order(self, **kwargs) -> Order:
        from datetime import datetime
        defaults = dict(
            order_id         = "ORD-001",
            symbol           = "SPY",
            side             = "buy",
            qty              = 10.0,
            order_type       = "market",
            status           = "pending",
            limit_price      = None,
            stop_price       = None,
            filled_qty       = 0.0,
            filled_avg_price = 0.0,
            time_in_force    = "day",
            client_order_id  = "CLT-001",
            submitted_at     = datetime.utcnow(),
            filled_at        = None,
        )
        defaults.update(kwargs)
        return Order(**defaults)

    def test_create_order(self):
        o = self._make_order()
        assert o.order_id == "ORD-001"

    def test_is_pending_true(self):
        o = self._make_order(status="pending")
        assert o.is_pending is True

    def test_is_filled_false_when_pending(self):
        o = self._make_order(status="pending")
        assert o.is_filled is False

    def test_is_filled_true(self):
        o = self._make_order(status="filled")
        assert o.is_filled is True

    def test_order_symbol(self):
        o = self._make_order(symbol="AAPL")
        assert o.symbol == "AAPL"

    def test_order_qty(self):
        o = self._make_order(qty=25.0)
        assert o.qty == 25.0

    def test_order_side_stored(self):
        o = self._make_order(side="buy")
        assert o.side in ("buy", "BUY", OrderSide.BUY.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Position dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionDataclass:

    def _make_pos(self, qty=100, avg_entry=450.0, current_price=460.0):
        return Position(
            symbol        = "SPY",
            qty           = qty,
            avg_entry     = avg_entry,
            market_value  = qty * current_price,
            unrealised_pl = qty * (current_price - avg_entry),
            current_price = current_price,
        )

    def test_is_long(self):
        pos = self._make_pos(qty=10)
        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.is_flat is False

    def test_is_short(self):
        pos = self._make_pos(qty=-10)
        assert pos.is_short is True
        assert pos.is_long is False
        assert pos.is_flat is False

    def test_is_flat(self):
        pos = self._make_pos(qty=0)
        assert pos.is_flat is True

    def test_unrealised_pl_positive(self):
        pos = self._make_pos(qty=10, avg_entry=450, current_price=460)
        assert pos.unrealised_pl > 0

    def test_unrealised_pl_negative(self):
        pos = self._make_pos(qty=10, avg_entry=460, current_price=450)
        assert pos.unrealised_pl < 0

    def test_market_value(self):
        pos = self._make_pos(qty=10, current_price=500)
        assert abs(pos.market_value - 5000) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PaperBroker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperBrokerExtended:

    def test_connect_and_name(self):
        b = _fresh_broker()
        assert isinstance(b.name, str)

    def test_is_paper_true(self):
        b = _fresh_broker()
        assert b.is_paper is True

    def test_initial_cash_correct(self):
        b = _fresh_broker(capital=50_000)
        acct = b.get_account()
        assert abs(acct["cash"] - 50_000) < 1.0

    def test_initial_equity_equals_initial_capital(self):
        b = _fresh_broker(capital=80_000)
        acct = b.get_account()
        assert abs(acct["equity"] - 80_000) < 1.0

    def test_get_positions_empty_initially(self):
        b = _fresh_broker()
        b.update_price("SPY", 450.0)
        positions = b.get_positions()
        assert isinstance(positions, list)
        assert len(positions) == 0

    def test_update_price(self):
        b = _fresh_broker()
        b.update_price("AAPL", 175.0)
        bar = b.get_latest_bar("AAPL")
        assert bar is not None

    def test_buy_order_reduces_cash(self):
        b = _fresh_broker(capital=100_000)
        b.update_price("SPY", 400.0)
        # submit_order(symbol, qty, side, order_type)
        b.submit_order("SPY", 10, OrderSide.BUY, OrderType.MARKET)
        acct = b.get_account()
        assert acct["cash"] < 100_000

    def test_buy_creates_position(self):
        b = _fresh_broker()
        b.update_price("SPY", 450.0)
        b.submit_order("SPY", 5, OrderSide.BUY, OrderType.MARKET)
        positions = b.get_positions()
        assert len(positions) == 1

    def test_position_qty_matches(self):
        b = _fresh_broker()
        b.update_price("MSFT", 300.0)
        b.submit_order("MSFT", 20, OrderSide.BUY, OrderType.MARKET)
        positions = b.get_positions()
        pos = next((p for p in positions if p.symbol == "MSFT"), None)
        assert pos is not None
        assert pos.qty == 20

    def test_get_fills(self):
        b = _fresh_broker()
        b.update_price("SPY", 400.0)
        b.submit_order("SPY", 3, OrderSide.BUY, OrderType.MARKET)
        fills = b.get_fills()
        assert isinstance(fills, list)
        assert len(fills) >= 1

    def test_fill_has_correct_symbol(self):
        b = _fresh_broker()
        b.update_price("AMZN", 150.0)
        b.submit_order("AMZN", 2, OrderSide.BUY, OrderType.MARKET)
        fills = b.get_fills()
        fill = next((f for f in fills if f.symbol == "AMZN"), None)
        assert fill is not None

    def test_list_orders_returns_list(self):
        b = _fresh_broker()
        b.update_price("SPY", 450.0)
        b.submit_order("SPY", 1, OrderSide.BUY, OrderType.MARKET)
        orders = b.list_orders()
        assert isinstance(orders, list)

    def test_slippage_applied_to_buy(self):
        """With slippage, fill price >= ask for buys."""
        b = PaperBroker(initial_capital=100_000, slippage=0.01)
        b.connect()
        b.update_price("SPY", 400.0)
        b.submit_order("SPY", 1, OrderSide.BUY, OrderType.MARKET)
        fills = b.get_fills()
        if fills:
            assert fills[0].price >= 400.0

    def test_commission_charged(self):
        b = PaperBroker(initial_capital=100_000, commission=1.0)
        b.connect()
        b.update_price("SPY", 400.0)
        b.submit_order("SPY", 1, OrderSide.BUY, OrderType.MARKET)
        acct = b.get_account()
        # cost = price + commission; cash should be less than 100_000 - 390
        assert acct["cash"] < 100_000 - 390

    def test_reset_clears_state(self):
        b = _fresh_broker()
        b.update_price("SPY", 450.0)
        b.submit_order("SPY", 5, OrderSide.BUY, OrderType.MARKET)
        b.reset()
        positions = b.get_positions()
        assert len(positions) == 0

    def test_disconnect_does_not_raise(self):
        b = _fresh_broker()
        b.disconnect()  # should not raise

    def test_update_prices_bulk(self):
        b = _fresh_broker()
        prices = {"SPY": 450.0, "AAPL": 175.0, "MSFT": 300.0}
        b.update_prices(prices)
        for sym in prices:
            bar = b.get_latest_bar(sym)
            assert bar is not None

    def test_sell_after_buy_updates_cash(self):
        b = _fresh_broker(capital=100_000)
        b.update_price("SPY", 400.0)
        b.submit_order("SPY", 10, OrderSide.BUY, OrderType.MARKET)
        cash_after_buy = b.get_account()["cash"]
        b.update_price("SPY", 410.0)
        b.submit_order("SPY", 10, OrderSide.SELL, OrderType.MARKET)
        cash_after_sell = b.get_account()["cash"]
        assert cash_after_sell > cash_after_buy

    def test_account_has_expected_keys(self):
        b = _fresh_broker()
        acct = b.get_account()
        assert "cash" in acct
        assert "equity" in acct

    def test_no_price_raises_or_returns_none(self):
        b = _fresh_broker()
        bar = b.get_latest_bar("UNKNOWN_TICKER_ZZZ")
        # Either None or an empty bar
        assert bar is None or isinstance(bar, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# LiveTradingLoop
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveTradingLoopExtended:

    def _make_loop(self, symbol="SPY", indicators=None) -> LiveTradingLoop:
        broker = _fresh_broker()
        # indicators is a dict: {name: {"enabled": bool, "params": dict}}
        if indicators is None:
            ind_dict = {"RSI": {"enabled": True, "params": {"period": 14}}}
        else:
            ind_dict = {name: {"enabled": True, "params": params}
                        for name, params in indicators}
        return LiveTradingLoop(
            broker           = broker,
            symbol           = symbol,
            indicators       = ind_dict,
            max_buffer_bars  = 60,
        )

    def test_loop_creation(self):
        loop = self._make_loop()
        assert loop is not None

    def test_seed_buffer(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        # buffer should be populated
        assert loop._ohlcv_buffer is not None
        assert len(loop._ohlcv_buffer) > 0

    def test_run_once_returns_live_run_result(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert isinstance(result, LiveRunResult)

    def test_run_once_result_has_symbol(self):
        loop = self._make_loop(symbol="AAPL")
        loop.seed_buffer(DF_60)
        loop.broker.update_price("AAPL", 175.0)
        result = loop.run_once()
        assert result.symbol == "AAPL"

    def test_run_once_result_has_timestamp(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert result.timestamp is not None

    def test_run_once_action_is_string(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert isinstance(result.action, str)

    def test_run_once_signal_is_numeric_or_none(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert isinstance(result.signal, (int, float, type(None)))

    def test_get_run_log_is_list(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        loop.run_once()
        log = loop.get_run_log()
        assert isinstance(log, list)
        assert len(log) >= 1

    def test_multiple_run_once_calls(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        broker = loop.broker
        for price in [450.0, 451.0, 449.0]:
            broker.update_price("SPY", price)
            result = loop.run_once()
            assert isinstance(result, LiveRunResult)

    def test_loop_with_multiple_indicators(self):
        loop = self._make_loop(indicators=[
            ("RSI", {"period": 14}),
            ("MACD", {}),
        ])
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert isinstance(result, LiveRunResult)

    def test_run_log_grows_per_call(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        loop.run_once()
        loop.broker.update_price("SPY", 451.0)
        loop.run_once()
        log = loop.get_run_log()
        assert len(log) == 2

    def test_live_run_result_has_account(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        # 'account' field should be set (dict or None)
        assert result.account is None or isinstance(result.account, dict)

    def test_live_run_result_reason_is_string(self):
        loop = self._make_loop()
        loop.seed_buffer(DF_60)
        loop.broker.update_price("SPY", 450.0)
        result = loop.run_once()
        assert isinstance(result.reason, str)
