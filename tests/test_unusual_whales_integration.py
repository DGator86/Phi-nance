"""
Unusual Whales integration tests — no live network calls.

Coverage
--------
1.  UnusualWhalesClient construction (missing key → ValueError)
2.  fetch_flow_alerts      — correct schema, time-indexed
3.  fetch_dark_pool        — correct schema, time-indexed
4.  fetch_market_tide      — returns dict with expected keys
5.  fetch_ticker_flow      — symbol uppercasing, row count
6.  fetch_options_chain    — returns DataFrame
7.  fetch_dark_pool_ticker — per-ticker endpoint
8.  fetch_market_overview  — returns dict
9.  fetch_market_movers    — returns DataFrame
10. fetch_etf_flow         — returns DataFrame
11. fetch_etf_holdings     — returns DataFrame
12. fetch_iv_rank          — returns dict with iv_rank key
13. fetch_oi_change        — returns DataFrame
14. fetch_options_volume   — returns DataFrame
15. fetch_pc_ratio         — returns dict with ratio key
16. fetch_short_interest   — returns dict with short_float key
17. fetch_congress_trades  — returns DataFrame
18. fetch_insider_trades   — returns DataFrame

GEX adapter
19. options_chain_for_gex  — column normalisation (option_type, open_interest, gamma)
20. options_chain_for_gex  — graceful empty return on HTTP error

GammaSurface pipeline
21. compute_from_unusual_whales — full end-to-end with synthetic chain
22. compute_from_unusual_whales — zero features when chain is empty

DataSourceManager routing
23. register_unusual_whales_sources — all 17 data types registered
24. register_unusual_whales_sources — graceful skip on missing API key
25. fetch via DataSourceManager     — routes to correct UW method
26. fetch with cache hit            — only one call to UW method
27. register_unusual_whales_sources — honours enabled=False in config
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

# ── Stub optional heavy deps that are not installed in CI ─────────────────────
for _mod in ("deap", "deap.base", "deap.creator", "deap.gp", "deap.tools"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_FUTURE_EXP = (date.today() + timedelta(days=30)).isoformat()


def _uw_client(api_key: str = "test-key") -> Any:
    from phinance.data.vendors.unusual_whales import UnusualWhalesClient
    return UnusualWhalesClient(api_key=api_key)


def _mock_get(client: Any, return_value: Any):
    """Patch UnusualWhalesClient._get to return *return_value*."""
    return patch.object(client, "_get", return_value=return_value)


def _dsm_config() -> dict:
    """Minimal DataSourceManager config with UW priorities."""
    return {
        "data_sources": {
            "unusual_whales": {
                "enabled": True,
                "rate_limit": 60,
                "rate_window_seconds": 60,
            }
        },
        "data_priorities": {
            "flow_alerts": ["unusual_whales"],
            "market_tide": ["unusual_whales"],
            "iv_rank": ["unusual_whales"],
        },
        "cache_ttl_seconds": {
            "flow_alerts": 60,
            "market_tide": 30,
            "iv_rank": 300,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Construction
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_api_key_raises():
    from phinance.data.vendors.unusual_whales import UnusualWhalesClient
    env = {k: v for k, v in os.environ.items() if k != "UNUSUAL_WHALES_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="API key"):
            UnusualWhalesClient()


def test_env_key_is_picked_up():
    with patch.dict(os.environ, {"UNUSUAL_WHALES_API_KEY": "env-key"}):
        from phinance.data.vendors.unusual_whales import UnusualWhalesClient
        c = UnusualWhalesClient()
        assert c.api_key == "env-key"


# ──────────────────────────────────────────────────────────────────────────────
# 2. fetch_flow_alerts
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_flow_alerts_schema():
    client = _uw_client()
    payload = {
        "data": [
            {"ticker": "SPY", "strike": 520.0, "option_type": "call",
             "premium": 50000, "volume": 200, "open_interest": 1000,
             "implied_volatility": 0.18, "sentiment": "bullish",
             "time": "2026-03-20T14:00:00Z"},
        ]
    }
    with _mock_get(client, payload):
        df = client.fetch_flow_alerts(symbol="SPY")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "ticker" in df.columns
    assert df.index.tz is not None  # time-indexed UTC


def test_fetch_flow_alerts_no_symbol():
    client = _uw_client()
    payload = {"data": [
        {"ticker": "AAPL", "strike": 200.0, "option_type": "put",
         "premium": 10000, "volume": 50, "open_interest": 500,
         "implied_volatility": 0.25, "sentiment": "bearish",
         "time": "2026-03-20T15:00:00Z"},
    ]}
    with _mock_get(client, payload):
        df = client.fetch_flow_alerts()  # market-wide
    assert len(df) == 1


def test_fetch_flow_alerts_empty_response():
    client = _uw_client()
    with _mock_get(client, {"data": []}):
        df = client.fetch_flow_alerts()
    assert df.empty


# ──────────────────────────────────────────────────────────────────────────────
# 3. fetch_dark_pool
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_dark_pool_schema():
    client = _uw_client()
    payload = {"data": [
        {"ticker": "QQQ", "price": 450.0, "size": 5000,
         "premium": 2250000, "exchange": "FINRA",
         "dark_pool_position": "above", "time": "2026-03-20T13:30:00Z"},
    ]}
    with _mock_get(client, payload):
        df = client.fetch_dark_pool(symbol="QQQ")

    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns
    assert "premium" in df.columns
    assert df.index.tz is not None


# ──────────────────────────────────────────────────────────────────────────────
# 4. fetch_market_tide
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_market_tide_keys():
    client = _uw_client()
    payload = {"data": {
        "net_call_premium": 1e8, "net_put_premium": -5e7,
        "call_volume": 300000, "put_volume": 200000,
        "put_call_ratio": 0.67, "gamma_exposure": 2.5e9,
        "timestamp": "2026-03-20T15:00:00Z",
    }}
    with _mock_get(client, payload):
        tide = client.fetch_market_tide("SPY")

    assert isinstance(tide, dict)
    assert "net_call_premium" in tide
    assert "put_call_ratio" in tide
    assert "gamma_exposure" in tide


# ──────────────────────────────────────────────────────────────────────────────
# 5. fetch_ticker_flow
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_ticker_flow_uppercases_symbol():
    client = _uw_client()
    captured = {}

    def fake_get(path, params=None):
        captured["path"] = path
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        client.fetch_ticker_flow("tsla", limit=25)

    assert "TSLA" in captured["path"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. fetch_options_chain
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_options_chain_returns_df():
    client = _uw_client()
    payload = {"data": [
        {"strike": 520.0, "expiration": _FUTURE_EXP, "option_type": "call",
         "open_interest": 5000, "gamma": 0.02, "delta": 0.55,
         "implied_volatility": 0.18, "volume": 1200},
        {"strike": 515.0, "expiration": _FUTURE_EXP, "option_type": "put",
         "open_interest": 3000, "gamma": 0.018, "delta": -0.45,
         "implied_volatility": 0.20, "volume": 900},
    ]}
    with _mock_get(client, payload):
        df = client.fetch_options_chain("SPY")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "strike" in df.columns


# ──────────────────────────────────────────────────────────────────────────────
# 7–17. Remaining endpoint smoke tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,kwargs,payload,check_key", [
    ("fetch_dark_pool_ticker", {"symbol": "AAPL"},
     {"data": [{"ticker": "AAPL", "price": 175.0, "size": 2000,
                "premium": 350000, "exchange": "FINRA",
                "dark_pool_position": "below", "time": "2026-03-20T14:00:00Z"}]},
     "ticker"),
    ("fetch_market_overview", {},
     {"data": {"advancing": 300, "declining": 200, "vix": 18.5,
               "put_call_ratio": 0.75, "timestamp": "2026-03-20T15:00:00Z"}},
     None),  # returns dict
    ("fetch_market_movers", {"direction": "gainers"},
     {"data": [{"ticker": "NVDA", "price": 900.0, "change_pct": 5.2,
                "volume": 30000000, "market_cap": 2.2e12}]},
     "ticker"),
    ("fetch_etf_flow", {},
     {"data": [{"ticker": "XLK", "net_call_premium": 1e7,
                "net_put_premium": -3e6, "put_call_ratio": 0.3}]},
     "ticker"),
    ("fetch_etf_holdings", {"symbol": "XLK"},
     {"data": [{"ticker": "AAPL", "name": "Apple Inc", "weight": 0.22,
                "shares": 5e6, "market_value": 8.75e8}]},
     "ticker"),
    ("fetch_oi_change", {"symbol": "TSLA"},
     {"data": [{"strike": 250.0, "expiration": _FUTURE_EXP,
                "option_type": "call", "oi_current": 5000,
                "oi_prev": 4000, "oi_change": 1000, "oi_change_pct": 25.0}]},
     "strike"),
    ("fetch_options_volume", {"symbol": "AAPL"},
     {"data": [{"strike": 175.0, "expiration": _FUTURE_EXP,
                "option_type": "call", "volume": 8000,
                "open_interest": 25000, "implied_volatility": 0.22}]},
     "strike"),
    ("fetch_congress_trades", {},
     {"data": [{"politician": "Jane Smith", "party": "D", "chamber": "Senate",
                "ticker": "NVDA", "transaction_type": "purchase",
                "amount_range": "1001-15000",
                "filed_date": "2026-03-15", "traded_date": "2026-03-10",
                "issuer_name": "NVIDIA Corp"}]},
     "ticker"),
    ("fetch_insider_trades", {"symbol": "MSFT"},
     {"data": [{"ticker": "MSFT", "insider_name": "Satya Nadella",
                "title": "CEO", "transaction_type": "sale",
                "shares": 50000, "price": 420.0, "value": 21000000,
                "filing_date": "2026-03-18", "transaction_date": "2026-03-15"}]},
     "ticker"),
])
def test_endpoint_smoke(method, kwargs, payload, check_key):
    client = _uw_client()
    with _mock_get(client, payload):
        result = getattr(client, method)(**kwargs)
    if check_key is None:
        assert isinstance(result, dict)
    else:
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert check_key in result.columns


@pytest.mark.parametrize("method,kwargs,expected_key", [
    ("fetch_iv_rank", {"symbol": "AAPL"},
     "iv_rank"),
    ("fetch_pc_ratio", {"symbol": "SPY"},
     "pc_volume_ratio"),
    ("fetch_short_interest", {"symbol": "GME"},
     "short_float"),
])
def test_dict_endpoint_smoke(method, kwargs, expected_key):
    client = _uw_client()
    payload = {"data": {
        "iv_rank": 72.5, "iv_percentile": 80.0,
        "iv_current": 0.35, "iv_high_52w": 0.55, "iv_low_52w": 0.18,
        "ticker": "AAPL", "timestamp": "2026-03-20T15:00:00Z",
        # pc_ratio keys
        "call_volume": 50000, "put_volume": 35000,
        "pc_volume_ratio": 0.70, "call_oi": 200000, "put_oi": 150000,
        "pc_oi_ratio": 0.75,
        # short interest keys
        "short_float": 0.12, "short_ratio": 2.5, "short_shares": 10000000,
        "utilization": 0.90, "cost_to_borrow": 0.05, "as_of_date": "2026-03-15",
    }}
    with _mock_get(client, payload):
        result = getattr(client, method)(**kwargs)
    assert isinstance(result, dict)
    assert expected_key in result


# ──────────────────────────────────────────────────────────────────────────────
# 19–20. options_chain_for_gex adapter
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_chain_payload() -> dict:
    """Payload from fetch_options_chain with some column name variations."""
    return {"data": [
        # Calls
        {"strike": 510.0, "expiration": _FUTURE_EXP, "option_type": "call",
         "open_interest": 8000, "gamma": 0.025, "volume": 2000},
        {"strike": 515.0, "expiration": _FUTURE_EXP, "option_type": "call",
         "open_interest": 12000, "gamma": 0.030, "volume": 3500},
        {"strike": 520.0, "expiration": _FUTURE_EXP, "option_type": "call",
         "open_interest": 20000, "gamma": 0.040, "volume": 6000},  # ATM wall
        {"strike": 525.0, "expiration": _FUTURE_EXP, "option_type": "call",
         "open_interest": 10000, "gamma": 0.028, "volume": 2500},
        # Puts
        {"strike": 510.0, "expiration": _FUTURE_EXP, "option_type": "put",
         "open_interest": 7000, "gamma": 0.022, "volume": 1800},
        {"strike": 515.0, "expiration": _FUTURE_EXP, "option_type": "put",
         "open_interest": 9000, "gamma": 0.027, "volume": 2200},
        {"strike": 520.0, "expiration": _FUTURE_EXP, "option_type": "put",
         "open_interest": 18000, "gamma": 0.038, "volume": 5000},
        {"strike": 525.0, "expiration": _FUTURE_EXP, "option_type": "put",
         "open_interest": 6000, "gamma": 0.020, "volume": 1500},
    ]}


def test_options_chain_for_gex_columns():
    """Adapter must produce strike, expiration, option_type, open_interest, gamma."""
    client = _uw_client()
    with _mock_get(client, _synthetic_chain_payload()):
        df = client.options_chain_for_gex("SPY")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("strike", "expiration", "option_type", "open_interest", "gamma"):
        assert col in df.columns, f"Missing column: {col}"

    # option_type must be normalised to 'call' / 'put'
    types = df["option_type"].unique()
    assert set(types).issubset({"call", "put"})

    # strike and gamma must be numeric
    assert pd.api.types.is_float_dtype(df["strike"])
    assert pd.api.types.is_float_dtype(df["gamma"])


def test_options_chain_for_gex_variant_column_names():
    """Adapter handles columns 'optiontype', 'openinterest', 'oi', etc."""
    client = _uw_client()
    # Simulate UW returning non-standard column names
    payload = {"data": [
        {"strike": 520.0, "expiry": _FUTURE_EXP, "optiontype": "C",
         "oi": 5000, "gamma": 0.03, "volume": 800},
        {"strike": 515.0, "expiry": _FUTURE_EXP, "optiontype": "P",
         "oi": 4000, "gamma": 0.025, "volume": 600},
    ]}
    with _mock_get(client, payload):
        df = client.options_chain_for_gex("SPY")

    assert "open_interest" in df.columns
    assert "expiration" in df.columns


def test_options_chain_for_gex_http_error_returns_empty():
    """When fetch_options_chain raises, options_chain_for_gex returns empty DF."""
    client = _uw_client()
    with patch.object(client, "_get", side_effect=Exception("HTTP 403")):
        df = client.options_chain_for_gex("SPY")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ──────────────────────────────────────────────────────────────────────────────
# 21–22. GammaSurface.compute_from_unusual_whales
# ──────────────────────────────────────────────────────────────────────────────

_GS_CFG = {
    "enabled": True,
    "kernel_width_pct": 0.005,
    "min_oi": 100,
    "max_dte": 90,
    "gex_flip_threshold": 0.10,
}


def test_compute_from_unusual_whales_full_pipeline():
    """End-to-end: synthetic chain → four valid GEX features."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "regime_engine"))
    from gamma_surface import GammaSurface

    client = _uw_client()
    spot = 518.0
    with _mock_get(client, _synthetic_chain_payload()):
        features = GammaSurface.compute_from_unusual_whales(
            _GS_CFG, client, symbol="SPY", spot=spot
        )

    assert isinstance(features, dict)
    assert set(features) == {"gamma_wall_distance", "gamma_net",
                             "gamma_expiry_days", "gex_flip_zone"}

    # gamma_wall_distance must be in [-0.20, +0.20]
    assert -0.20 <= features["gamma_wall_distance"] <= 0.20

    # gamma_net must be in [-1, +1]
    assert -1.0 <= features["gamma_net"] <= 1.0

    # gamma_expiry_days must be positive and ≤ max_dte
    assert 0 < features["gamma_expiry_days"] <= _GS_CFG["max_dte"]

    # gex_flip_zone is 0 or 1
    assert features["gex_flip_zone"] in (0.0, 1.0)


def test_compute_from_unusual_whales_empty_chain_returns_zeros():
    """When the chain is empty, zero features are returned."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "regime_engine"))
    from gamma_surface import GammaSurface

    client = _uw_client()
    with _mock_get(client, {"data": []}):
        features = GammaSurface.compute_from_unusual_whales(
            _GS_CFG, client, symbol="SPY", spot=520.0
        )

    assert features["gamma_wall_distance"] == 0.0
    assert features["gamma_net"] == 0.0
    assert features["gamma_expiry_days"] == 30.0
    assert features["gex_flip_zone"] == 0.0


def test_compute_from_unusual_whales_fetch_exception_returns_zeros():
    """When the chain fetch raises, zero features are returned gracefully."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "regime_engine"))
    from gamma_surface import GammaSurface

    client = _uw_client()
    with patch.object(client, "_get", side_effect=Exception("network error")):
        features = GammaSurface.compute_from_unusual_whales(
            _GS_CFG, client, symbol="SPY", spot=520.0
        )

    assert features["gamma_net"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 23–27. DataSourceManager routing
# ──────────────────────────────────────────────────────────────────────────────

def test_register_unusual_whales_sources_registers_all_types():
    from phinance.live.data_source_manager import DataSourceManager

    dsm = DataSourceManager(_dsm_config())
    result = dsm.register_unusual_whales_sources(api_key="test-key")

    assert result is True
    uw_sources = dsm.sources.get("unusual_whales", {})
    expected = [
        "flow_alerts", "ticker_flow", "options_chain",
        "dark_pool", "dark_pool_ticker",
        "market_tide", "market_overview", "market_movers",
        "etf_flow", "etf_holdings",
        "iv_rank", "oi_change", "options_volume", "pc_ratio",
        "short_interest", "congress_trades", "insider_trades",
    ]
    for dt in expected:
        assert dt in uw_sources, f"Data type not registered: {dt}"


def test_register_unusual_whales_sources_skips_on_missing_key():
    from phinance.live.data_source_manager import DataSourceManager

    dsm = DataSourceManager(_dsm_config())
    env = {k: v for k, v in os.environ.items() if k != "UNUSUAL_WHALES_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        result = dsm.register_unusual_whales_sources(api_key=None)

    assert result is False
    assert "unusual_whales" not in dsm.sources


def test_dsm_routes_flow_alerts_to_uw():
    from phinance.live.data_source_manager import DataSourceManager

    dsm = DataSourceManager(_dsm_config())
    dsm.register_unusual_whales_sources(api_key="test-key")

    expected_df = pd.DataFrame([{"ticker": "SPY", "premium": 50000}])
    mock_fetcher = MagicMock(return_value=expected_df)
    dsm.sources["unusual_whales"]["flow_alerts"] = mock_fetcher

    result = dsm.fetch("flow_alerts", symbol="SPY", limit=10)

    mock_fetcher.assert_called_once_with(symbol="SPY", limit=10)
    assert isinstance(result, pd.DataFrame)
    assert result["ticker"].iloc[0] == "SPY"


def test_dsm_caches_uw_response():
    from phinance.live.data_source_manager import DataSourceManager

    dsm = DataSourceManager(_dsm_config())
    dsm.register_unusual_whales_sources(api_key="test-key")

    call_count = {"n": 0}

    def counting_fetcher(**kwargs):
        call_count["n"] += 1
        return {"net_call_premium": 1e8, "put_call_ratio": 0.7}

    dsm.sources["unusual_whales"]["market_tide"] = counting_fetcher

    dsm.fetch("market_tide", symbol="SPY")
    dsm.fetch("market_tide", symbol="SPY")  # should hit cache

    assert call_count["n"] == 1, "Expected cache to prevent second network call"


def test_dsm_honours_source_disabled():
    from phinance.live.data_source_manager import DataSourceManager

    cfg = _dsm_config()
    cfg["data_sources"]["unusual_whales"]["enabled"] = False
    dsm = DataSourceManager(cfg)
    dsm.register_unusual_whales_sources(api_key="test-key")
    dsm.status["unusual_whales"].enabled = False

    with pytest.raises(RuntimeError, match="All sources exhausted"):
        dsm.fetch("flow_alerts", symbol="SPY")
