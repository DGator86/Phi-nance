from __future__ import annotations

from phinance.live.data_source_manager import DataSourceManager


def _config():
    return {
        "data_sources": {
            "primary": {"enabled": True, "rate_limit": 1, "rate_window_seconds": 60},
            "fallback": {"enabled": True, "rate_limit": 10, "rate_window_seconds": 60},
        },
        "data_priorities": {"bars": ["primary", "fallback"]},
        "cache_ttl_seconds": {"bars": 30},
    }


def test_fallback_is_used_when_primary_throws():
    manager = DataSourceManager(_config())
    manager.register_source("primary", "bars", lambda **_: (_ for _ in ()).throw(RuntimeError("limited")))
    manager.register_source("fallback", "bars", lambda **_: {"source": "fallback"})

    payload = manager.fetch("bars", symbol="SPY")
    assert payload["source"] == "fallback"


def test_cache_avoids_redundant_calls():
    manager = DataSourceManager(_config())
    calls = {"count": 0}

    def fetcher(**_):
        calls["count"] += 1
        return {"value": 1}

    manager.register_source("primary", "bars", fetcher)
    manager.register_source("fallback", "bars", lambda **_: {"value": 2})

    manager.fetch("bars", symbol="SPY")
    manager.fetch("bars", symbol="SPY")
    assert calls["count"] == 1
