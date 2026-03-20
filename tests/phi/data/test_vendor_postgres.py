from __future__ import annotations

import pandas as pd
import pytest

from phi.data.vendor_postgres import PostgresOptionsVendor


class _StubEngine:
    pass


def test_safe_table_name_strips_unsafe_chars():
    assert PostgresOptionsVendor._safe_table_name("SPY;DROP TABLE") == "spydroptable"


def test_safe_identifier_rejects_empty():
    with pytest.raises(ValueError, match="timestamp_column"):
        PostgresOptionsVendor._safe_identifier("$%^", "timestamp_column")


def test_fetch_rejects_unknown_timestamp_unit(monkeypatch):
    vendor = PostgresOptionsVendor.__new__(PostgresOptionsVendor)
    vendor.engine = _StubEngine()

    with pytest.raises(ValueError, match="timestamp_unit"):
        vendor.fetch("SPY", "2023-01-01", "2023-01-02", timestamp_unit="minutes")


def test_fetch_passes_expected_sql_params(monkeypatch):
    vendor = PostgresOptionsVendor.__new__(PostgresOptionsVendor)
    vendor.engine = _StubEngine()

    captured = {}

    def fake_read_sql(query, engine, params):
        captured["query"] = str(query)
        captured["engine"] = engine
        captured["params"] = params
        return pd.DataFrame({"quote_time": [1_672_531_200_000], "last": [1.23]})

    monkeypatch.setattr("phi.data.vendor_postgres.pd.read_sql", fake_read_sql)

    out = vendor.fetch(
        "SPY",
        "2023-01-01",
        "2023-01-02",
        timestamp_column="quote_time",
        timestamp_unit="ms",
    )

    assert not out.empty
    assert captured["engine"] is vendor.engine
    assert captured["params"]["start_ts"] == 1_672_531_200_000
    assert captured["params"]["end_ts"] == 1_672_617_600_000
