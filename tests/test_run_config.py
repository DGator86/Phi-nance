"""Unit tests for phi/run_config.py."""

from __future__ import annotations

from phi.run_config import RunConfig


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.symbols == ["SPY"]
    assert cfg.timeframe == "1D"
    assert cfg.vendor == "alphavantage"
    assert cfg.initial_capital == 100_000.0
    assert cfg.trading_mode == "equities"
    assert cfg.blend_method == "weighted_sum"
    assert cfg.phiai_enabled is False


def test_runconfig_to_dict_roundtrip():
    cfg = RunConfig(
        dataset_id="test-123",
        symbols=["AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2023-01-01",
        initial_capital=50_000.0,
    )
    d = cfg.to_dict()
    cfg2 = RunConfig.from_dict(d)
    assert cfg2.dataset_id == "test-123"
    assert cfg2.symbols == ["AAPL", "MSFT"]
    assert cfg2.start_date == "2022-01-01"
    assert cfg2.end_date == "2023-01-01"
    assert cfg2.initial_capital == 50_000.0


def test_runconfig_custom_symbols():
    cfg = RunConfig(symbols=["BTC", "ETH", "SOL"])
    assert cfg.symbols == ["BTC", "ETH", "SOL"]
    assert len(cfg.symbols) == 3
