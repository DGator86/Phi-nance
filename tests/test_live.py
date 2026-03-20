from __future__ import annotations

from datetime import datetime

from phi.live.broker import Broker, BrokerAccount, BrokerOrder, BrokerPosition
from phi.live.engine import LiveEngine
from phi.live.portfolio import LivePortfolio
from phi.live.risk import RiskLimits, RiskManager
from phi.live.strategy import LiveStrategy


class MockBroker(Broker):
    def __init__(self):
        self.orders: list[BrokerOrder] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=100000.0, equity=100000.0, buying_power=200000.0)

    def get_positions(self) -> list[BrokerPosition]:
        return []

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.orders)

    def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str):
        raise NotImplementedError

    def subscribe_bars(self, symbol: str, callback):
        callback({"symbol": symbol, "close": 100.0})

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self.orders.append(order)
        order.status = "filled"
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True


def test_portfolio_updates() -> None:
    p = LivePortfolio(initial_cash=1000)
    p.apply_fill("SPY", qty=2, price=100, side="buy")
    p.update_price("SPY", 110)
    assert p.equity() == 1020


def test_risk_rejects_oversized_order() -> None:
    rm = RiskManager(RiskLimits(max_position_pct=0.1))
    ok = rm.validate_order(BrokerOrder(symbol="SPY", qty=2, side="buy"), equity=1000, price=40)
    bad = rm.validate_order(BrokerOrder(symbol="SPY", qty=5, side="buy"), equity=1000, price=40)
    assert ok is True
    assert bad is False


def test_strategy_generates_order_after_price_change() -> None:
    broker = MockBroker()
    portfolio = LivePortfolio(initial_cash=10000)
    strat = LiveStrategy(config={"allocation_strategy": "equal_weight"}, broker=broker, portfolio=portfolio)
    orders = strat.on_bar({"symbol": "SPY", "close": 100})
    assert len(orders) == 1
    orders2 = strat.on_bar({"symbol": "SPY", "close": 101})
    assert isinstance(orders2, list)


def test_engine_loop_places_orders() -> None:
    broker = MockBroker()
    portfolio = LivePortfolio(initial_cash=10000)
    strategy = LiveStrategy(config={"allocation_strategy": "equal_weight"}, broker=broker, portfolio=portfolio)
    engine = LiveEngine(
        broker=broker,
        portfolio=portfolio,
        strategy=strategy,
        risk_manager=RiskManager(),
        symbols=["SPY"],
        update_interval=0,
    )
    engine.run(max_cycles=2)
    assert broker.connected is False
    assert len(engine.portfolio.equity_curve) >= 1
