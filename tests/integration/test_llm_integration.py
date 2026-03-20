from __future__ import annotations

from phinance.live.engine import LiveEngine
from phinance.live.order_manager import OrderManager


class DummyBroker:
    def safe_call(self, method_name: str, **kwargs):
        return {"method": method_name, **kwargs}


class DummyDataManager:
    def fetch(self, kind: str, symbol: str):
        assert kind == "quotes"
        return {"symbol": symbol, "price": 100.0}


class StubAdvisor:
    def explain_trades(self, trades, market_context):
        return f"Explained {market_context['symbol']} ({len(trades)} trades)"


def test_live_engine_advisor_manual_request():
    engine = LiveEngine(data_manager=DummyDataManager(), order_manager=OrderManager(DummyBroker()))
    engine.set_advisor(StubAdvisor())
    report = engine.request_advisor_report(symbol="SPY", quote={"price": 100.0})
    assert report is not None
    assert "Explained SPY" in report
    assert engine.last_advisor_report == report
