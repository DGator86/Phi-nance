"""Unit tests for phi/run_config.py."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from phi.run_config import RunConfig


def _valid_payload(**overrides):
    payload = {
        "dataset_id": "demo",
        "symbols": ["spy", "qqq"],
        "start_date": date(2022, 1, 1),
        "end_date": date(2022, 12, 31),
        "timeframe": "1D",
        "vendor": "alphavantage",
        "initial_capital": 100_000.0,
        "trading_mode": "equities",
        "indicators": {
            "RSI": {"enabled": True, "params": {"period": 14}},
            "MACD": {"enabled": False, "params": {}},
        },
        "blend_method": "weighted_sum",
        "blend_weights": {"RSI": 1.0},
        "phiai_enabled": False,
        "evaluation_metric": "roi",
    }
    payload.update(overrides)
    return payload


def test_valid_config_creation():
    cfg = RunConfig(**_valid_payload())
    assert cfg.symbols == ["SPY", "QQQ"]
    assert cfg.schema_version == 1


def test_invalid_date_range_raises():
    with pytest.raises(ValidationError):
        RunConfig(**_valid_payload(start_date=date(2023, 1, 1), end_date=date(2022, 1, 1)))


def test_invalid_symbols_raise():
    with pytest.raises(ValidationError):
        RunConfig(**_valid_payload(symbols=[]))

    with pytest.raises(ValidationError):
        RunConfig(**_valid_payload(symbols=["SPY", "spy"]))


def test_blend_weights_must_match_enabled_indicators():
    with pytest.raises(ValidationError):
        RunConfig(**_valid_payload(blend_weights={"MACD": 1.0}))


def test_blend_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        RunConfig(**_valid_payload(blend_weights={"RSI": 0.9}))


def test_save_load_round_trip(tmp_path):
    cfg = RunConfig(**_valid_payload())
    run_dir = tmp_path / "run1"
    cfg.save(run_dir)

    loaded = RunConfig.load(run_dir)
    assert loaded == cfg
    assert loaded.model_dump()["schema_version"] == 1


def test_migration_from_v0_config(tmp_path):
    run_dir = tmp_path / "run_v0"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _valid_payload()
    payload.pop("schema_version", None)
    payload["start"] = payload.pop("start_date").isoformat()
    payload["end"] = payload.pop("end_date").isoformat()

    config_path = run_dir / "config.json"
    config_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    loaded = RunConfig.from_json(config_path)
    assert loaded.schema_version == 1
    assert loaded.start_date == date(2022, 1, 1)
    assert loaded.end_date == date(2022, 12, 31)
