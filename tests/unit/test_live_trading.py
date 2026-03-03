"""
tests.unit.test_live_trading
==============================

Comprehensive unit tests for the phinance.live package:
  • Order / Fill / Position dataclasses (order_models)
  • PaperBroker (paper_engine)
  • BrokerAdapter abstract interface
  • LiveTradingLoop (trading_loop)

All tests run fully in-process — no network, no broker API keys required.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from tests.fixtures.ohlcv import make_ohlcv

# ── Helpers ────────────────────────────────────────────────────────────────────

DF200 = make_ohlcv(n=200)


# ═══════════════════════════════════════════════════════════════════════════════
# Order / Fill / Position dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderModels:

    def test_order_is_filled_property(self):
        from phinance.live.order_models import Order
        o = Order(order_id="1", symbol="SPY", side="buy", qty=10, status="filled")
        assert o.is_filled is True
        assert o.is_pending is False

    def test_order_is_pending_property(self):
        from phinance.live.order_models import Order
        o = Order(order_id="2", symbol="AAPL", side="sell", qty=5, status="submitted")
        assert o.is_pending is True
        assert o.is_filled is False

    def test_order_repr(self):
        from phinance.live.order_models import Order
        o = Order(order_id="3", symbol="MSFT", side="buy", qty=20)
        r = repr(o)
        assert "MSFT" in r
        assert "buy" in r

    def test_fill_notional(self):
        from phinance.live.order_models import Fill
        f = Fill(order_id="4", symbol="SPY", side="buy", qty=10, price=450.0)
        assert f.notional == pytest.approx(4500.0)

    def test_position_long_short_flat(self):
        from phinance.live.order_models import Position
        long_pos  = Position(symbol="SPY",  qty=10,  avg_entry=450.0)
        short_pos = Position(symbol="TSLA", qty=-5,  avg_entry=200.0)
        flat_pos  = Position(symbol="GOOG", qty=0,   avg_entry=100.0)

        assert long_pos.is_long  and not long_pos.is_short  and not long_pos.is_flat
        assert short_pos.is_short and not short_pos.is_long  and not short_pos.is_flat
        assert flat_pos.is_flat   and not flat_pos.is_long   and not flat_pos.is_short

    def test_position_repr(self):
        from phinance.live.order_models import Position
        p = Position(symbol="SPY", qty=10, avg_entry=450.0)
        assert "SPY" in repr(p)


# ═══════════════════════════════════════════════════════════════════════════════
# BrokerAdapter abstract interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrokerAdapterInterface:

    def test_abstract_cannot_instantiate(self):
        from phinance.live.broker_base import BrokerAdapter
        with pytest.raises(TypeError):
            BrokerAdapter()

    def test_order_side_enum_values(self):
        from phinance.live.broker_base import OrderSide
        assert OrderSide.BUY.value  == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_enum_values(self):
        from phinance.live.broker_base import OrderType
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value  == "limit"
        assert OrderType.STOP.value   == "stop"

    def test_order_status_enum_values(self):
        from phinance.live.broker_base import OrderStatus
        expected = {"pending", "submitted", "filled", "cancelled", "rejected"}
        actual = {s.value for s in OrderStatus}
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════════
# PaperBroker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperBrokerLifecycle:

    def test_connect_and_name(self):
        from phinance.live.paper_engine import PaperBroker
        broker = PaperBroker(initial_capital=50_000)
        broker.connect()
        assert broker.name == "PaperBroker"
        assert broker.is_paper is True

    def test_context_manager(self):
        from phinance.live.paper_engine import PaperBroker
        with PaperBroker(initial_capital=10_000) as broker:
            account = broker.get_account()
        assert account["cash"] == pytest.approx(10_000.0)

    def test_initial_account_state(self):
        from phinance.live.paper_engine import PaperBroker
        broker = PaperBroker(initial_capital=100_000)
        broker.connect()
        acc = broker.get_account()
        assert acc["equity"] == pytest.approx(100_000.0)
        assert acc["cash"]   == pytest.approx(100_000.0)

    def test_no_positions_initially(self):
        from phinance.live.paper_engine import PaperBroker
        broker = PaperBroker(initial_capital=50_000)
        broker.connect()
        assert broker.get_positions() == []

    def test_repr(self):
        from phinance.live.paper_engine import PaperBroker
        broker = PaperBroker()
        assert "PaperBroker" in repr(broker)
        assert "paper=True" in repr(broker)


class TestPaperBrokerOrderExecution:

    @pytest.fixture
    def broker(self):
        from phinance.live.paper_engine import PaperBroker
        b = PaperBroker(initial_capital=100_000, slippage=0.0, commission=0.0)
        b.connect()
        b.update_price("SPY", 450.0)
        return b

    def test_buy_market_order_fills(self, broker):
        from phinance.live.broker_base import OrderSide
        order = broker.submit_order("SPY", 10, OrderSide.BUY)
        assert order.is_filled
        assert order.filled_qty == pytest.approx(10.0)
        assert order.filled_avg_price == pytest.approx(450.0)

    def test_sell_market_order_fills(self, broker):
        from phinance.live.broker_base import OrderSide
        # Buy first
        broker.submit_order("SPY", 20, OrderSide.BUY)
        # Then sell
        order = broker.submit_order("SPY", 10, OrderSide.SELL)
        assert order.is_filled
        assert order.filled_qty == pytest.approx(10.0)

    def test_buy_reduces_cash(self, broker):
        from phinance.live.broker_base import OrderSide
        before = broker.get_account()["cash"]
        broker.submit_order("SPY", 10, OrderSide.BUY)
        after = broker.get_account()["cash"]
        assert after < before

    def test_sell_increases_cash(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.submit_order("SPY", 10, OrderSide.BUY)
        cash_after_buy = broker.get_account()["cash"]
        broker.submit_order("SPY", 10, OrderSide.SELL)
        cash_after_sell = broker.get_account()["cash"]
        assert cash_after_sell > cash_after_buy

    def test_position_created_after_buy(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.submit_order("SPY", 5, OrderSide.BUY)
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "SPY"
        assert positions[0].qty == pytest.approx(5.0)

    def test_get_order_by_id(self, broker):
        from phinance.live.broker_base import OrderSide
        order = broker.submit_order("SPY", 3, OrderSide.BUY)
        fetched = broker.get_order(order.order_id)
        assert fetched is not None
        assert fetched.order_id == order.order_id

    def test_list_orders(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.submit_order("SPY", 2, OrderSide.BUY)
        broker.submit_order("SPY", 3, OrderSide.BUY)
        orders = broker.list_orders()
        assert len(orders) >= 2

    def test_cancel_order_nonexistent_returns_false(self, broker):
        result = broker.cancel_order("nonexistent-id-xyz")
        assert result is False

    def test_large_order_consumes_all_cash(self, broker):
        """A very large order should consume most/all available cash."""
        from phinance.live.broker_base import OrderSide
        # Calculate max affordable qty
        account = broker.get_account()
        cash = account["cash"]
        max_qty = int(cash / 450.0)
        order = broker.submit_order("SPY", max_qty, OrderSide.BUY)
        assert order.is_filled
        after = broker.get_account()
        assert after["cash"] < cash

    def test_latest_bar_returns_series(self, broker):
        bar = broker.get_latest_bar("SPY")
        assert bar is not None
        assert isinstance(bar, pd.Series)
        assert "close" in bar.index

    def test_latest_bar_unknown_symbol_returns_none(self, broker):
        bar = broker.get_latest_bar("UNKNOWNTICKER")
        assert bar is None

    def test_update_prices_dict(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.update_prices({"SPY": 460.0, "AAPL": 185.0})
        bar = broker.get_latest_bar("AAPL")
        assert bar is not None
        assert bar["close"] == pytest.approx(185.0)

    def test_fills_list_populated_after_orders(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.submit_order("SPY", 5, OrderSide.BUY)
        fills = broker.get_fills()
        assert len(fills) >= 1

    def test_reset_clears_state(self, broker):
        from phinance.live.broker_base import OrderSide
        broker.submit_order("SPY", 5, OrderSide.BUY)
        broker.reset()
        assert broker.get_positions() == []
        assert broker.get_fills()     == []


class TestPaperBrokerSlippageCommission:

    def test_slippage_applied_on_buy(self):
        from phinance.live.paper_engine import PaperBroker
        from phinance.live.broker_base import OrderSide
        broker = PaperBroker(initial_capital=100_000, slippage=0.01, commission=0.0)
        broker.connect()
        broker.update_price("SPY", 400.0)
        order = broker.submit_order("SPY", 1, OrderSide.BUY)
        # With slippage 1% fill price should be slightly above 400
        assert order.filled_avg_price > 400.0

    def test_commission_applied(self):
        from phinance.live.paper_engine import PaperBroker
        from phinance.live.broker_base import OrderSide
        broker = PaperBroker(initial_capital=100_000, slippage=0.0, commission=1.0)
        broker.connect()
        broker.update_price("SPY", 400.0)
        before = broker.get_account()["cash"]
        broker.submit_order("SPY", 10, OrderSide.BUY)
        after = broker.get_account()["cash"]
        # Cash reduced by price*qty + commission*qty
        expected_cost = 400.0 * 10 + 1.0 * 10
        assert before - after == pytest.approx(expected_cost, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# LiveTradingLoop
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveTradingLoop:

    @pytest.fixture
    def loop_with_broker(self):
        from phinance.live.paper_engine import PaperBroker
        from phinance.live.trading_loop import LiveTradingLoop
        broker = PaperBroker(initial_capital=50_000, slippage=0.0, commission=0.0)
        broker.connect()
        broker.update_price("SPY", float(DF200["close"].iloc[-1]))
        loop = LiveTradingLoop(
            broker=broker,
            symbol="SPY",
            indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
            capital=50_000,
            signal_threshold=0.3,
            position_size_pct=0.1,
        )
        return loop, broker

    def test_run_once_returns_result(self, loop_with_broker):
        from phinance.live.trading_loop import LiveRunResult
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        assert isinstance(result, LiveRunResult)

    def test_result_has_signal(self, loop_with_broker):
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        assert isinstance(result.signal, float)
        assert -1.0 <= result.signal <= 1.0

    def test_result_action_is_valid(self, loop_with_broker):
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        assert result.action in ("buy", "sell", "hold")

    def test_result_symbol_matches(self, loop_with_broker):
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        assert result.symbol == "SPY"

    def test_result_str_representation(self, loop_with_broker):
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        s = str(result)
        assert "SPY" in s
        assert "signal" in s

    def test_seed_buffer_works(self, loop_with_broker):
        loop, _ = loop_with_broker
        loop.seed_buffer(DF200)
        result = loop.run_once(DF200)
        assert result is not None

    def test_run_log_appended(self, loop_with_broker):
        loop, _ = loop_with_broker
        loop.run_once(DF200)
        loop.run_once(DF200)
        log = loop.get_run_log()
        assert len(log) >= 2

    def test_multiple_indicators(self):
        from phinance.live.paper_engine import PaperBroker
        from phinance.live.trading_loop import LiveTradingLoop
        broker = PaperBroker(initial_capital=50_000, slippage=0.0, commission=0.0)
        broker.connect()
        broker.update_price("SPY", float(DF200["close"].iloc[-1]))
        loop = LiveTradingLoop(
            broker=broker,
            symbol="SPY",
            indicators={
                "RSI": {"enabled": True, "params": {}},
                "MACD": {"enabled": True, "params": {}},
            },
            capital=50_000,
        )
        result = loop.run_once(DF200)
        assert result is not None

    def test_insufficient_data_returns_hold(self):
        from phinance.live.paper_engine import PaperBroker
        from phinance.live.trading_loop import LiveTradingLoop
        broker = PaperBroker(initial_capital=50_000)
        broker.connect()
        broker.update_price("SPY", 100.0)
        loop = LiveTradingLoop(
            broker=broker,
            symbol="SPY",
            indicators={"RSI": {"enabled": True, "params": {}}},
            capital=50_000,
        )
        tiny_df = make_ohlcv(n=3)
        result = loop.run_once(tiny_df)
        assert result.action == "hold"

    def test_hold_zone_no_order(self, loop_with_broker):
        """If signal is within hold threshold, no order should be placed."""
        loop, _ = loop_with_broker
        result = loop.run_once(DF200)
        # If action is hold, order should be None
        if result.action == "hold":
            assert result.order is None


# ═══════════════════════════════════════════════════════════════════════════════
# AlpacaBroker / IBKRBroker — import-only tests (no live API)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlpacaBrokerImport:

    def test_alpaca_module_importable(self):
        from phinance.live import alpaca  # noqa: F401

    def test_alpaca_broker_class_exists(self):
        from phinance.live.alpaca import AlpacaBroker
        assert AlpacaBroker is not None

    def test_alpaca_raises_import_error_without_library(self):
        """AlpacaBroker raises ImportError when alpaca-py is not installed."""
        from phinance.live.alpaca import AlpacaBroker, ALPACA_AVAILABLE
        if not ALPACA_AVAILABLE:
            with pytest.raises(ImportError):
                AlpacaBroker(api_key="fake", secret_key="fake")
        else:
            pytest.skip("alpaca-py is installed; skip unavailability test")

    def test_alpaca_available_flag_is_bool(self):
        import phinance.live.alpaca as alp_mod
        assert isinstance(alp_mod.ALPACA_AVAILABLE, bool)


class TestIBKRBrokerImport:

    def test_ibkr_module_importable(self):
        from phinance.live import ibkr  # noqa: F401

    def test_ibkr_broker_class_exists(self):
        from phinance.live.ibkr import IBKRBroker
        assert IBKRBroker is not None

    def test_ibkr_raises_import_error_without_library(self):
        """IBKRBroker raises ImportError when ib_insync is not installed."""
        from phinance.live.ibkr import IBKRBroker, IB_AVAILABLE
        if not IB_AVAILABLE:
            with pytest.raises(ImportError):
                IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        else:
            pytest.skip("ib_insync is installed; skip unavailability test")

    def test_ibkr_available_flag_is_bool(self):
        import phinance.live.ibkr as ibkr_mod
        assert isinstance(ibkr_mod.IB_AVAILABLE, bool)
