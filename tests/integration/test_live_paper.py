from __future__ import annotations

from phinance.live.data_source_manager import DataSourceManager
from phinance.live.engine import LiveEngine
from phinance.live.order_manager import OrderManager


class DummyBroker:
    def safe_call(self, method_name: str, **kwargs):
        return {"method": method_name, **kwargs}


def test_live_engine_tick_with_mock_quote_source():
    cfg = {
        "data_sources": {"primary": {"enabled": True, "rate_limit": 10, "rate_window_seconds": 60}},
        "data_priorities": {"quotes": ["primary"]},
    }
    manager = DataSourceManager(cfg)
    manager.register_source("primary", "quotes", lambda symbol: {"symbol": symbol, "price": 100.0})

    order_manager = OrderManager(DummyBroker())
    engine = LiveEngine(data_manager=manager, order_manager=order_manager)
    engine.start()

    result = engine.tick("SPY")
    assert result["status"] == "ok"
    assert result["quote"]["symbol"] == "SPY"
