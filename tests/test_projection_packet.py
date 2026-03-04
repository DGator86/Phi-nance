"""ProjectionPacket: schema, versioning, fixtures, stub generation."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from phinence.contracts.projection_packet import (
    Horizon,
    ProjectionPacket,
    SchemaVersion,
    make_stub_packet,
    make_empty_horizon,
)


def test_stub_packet_has_all_required_fields() -> None:
    p = make_stub_packet("SPY")
    assert p.ticker == "SPY"
    assert p.schema_version == SchemaVersion.V1
    assert len(p.horizons) >= 1
    for hp in p.horizons:
        assert hp.horizon in Horizon
        assert 0 <= hp.confidence <= 1
        assert hp.direction.up + hp.direction.down + hp.direction.flat == pytest.approx(1.0)
        assert hp.cone.p50_bps <= hp.cone.p75_bps <= hp.cone.p90_bps


def test_empty_horizon_confidence_zero() -> None:
    hp = make_empty_horizon(Horizon.DAILY)
    assert hp.confidence == 0.0
    assert hp.horizon == Horizon.DAILY


def test_get_horizon() -> None:
    p = make_stub_packet("QQQ")
    h1m = p.get_horizon(Horizon.INTRADAY_1M)
    assert h1m is not None
    assert h1m.horizon == Horizon.INTRADAY_1M
    assert p.get_horizon(Horizon.INTRADAY_5M) is not None


def test_fixture_roundtrip() -> None:
    # Repo root is parent of tests/
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "projection_packet_sample.json"
    if not fixture_path.exists():
        pytest.skip("fixtures/projection_packet_sample.json not found")
    raw = json.loads(fixture_path.read_text())
    p = ProjectionPacket.model_validate(raw)
    assert p.ticker == "SPY"
    assert len(p.horizons) == 3
    # Serialize back
    out = p.model_dump(mode="json")
    assert "schema_version" in out
    assert out["ticker"] == "SPY"


def test_valid_packet_with_zero_confidence() -> None:
    """Done when: valid ProjectionPacket with confidences 0.0 when data missing."""
    p = make_stub_packet("AAPL", as_of=datetime(2024, 6, 1, 12, 0, 0))
    for hp in p.horizons:
        assert hp.confidence == 0.0
    assert p.model_dump(mode="json")  # must serialize
