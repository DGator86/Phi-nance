"""QuantConnect bundle export (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phi.integrations.quantconnect import MANIFEST_VERSION, write_quantconnect_bundle


def test_write_quantconnect_bundle_minimal(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1e6, 1.1e6, 1.2e6],
        },
        index=idx,
    )
    out = write_quantconnect_bundle(
        tmp_path / "bundle",
        symbol="TEST",
        timeframe="1D",
        start="2024-01-02",
        end="2024-01-04",
        ohlcv_vendor="test",
        df=df,
    )
    assert (out / "ohlcv.csv").exists()
    assert (out / "manifest.json").exists()
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["schema_version"] == MANIFEST_VERSION
    assert man["symbol"] == "TEST"
    assert man["rows"] == 3
    assert "signal_card" not in man["files"]

    csv = (out / "ohlcv.csv").read_text(encoding="utf-8")
    assert "time,open,high,low,close,volume" in csv.splitlines()[0]


def test_write_quantconnect_bundle_with_signal_card(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100.0, 200.0],
        },
        index=idx,
    )
    write_quantconnect_bundle(
        tmp_path / "b2",
        symbol="X",
        timeframe="1D",
        start="2024-01-02",
        end="2024-01-03",
        ohlcv_vendor="yfinance",
        df=df,
        signal_card={"symbol": "X", "action": "WAIT"},
    )
    man = json.loads((tmp_path / "b2" / "manifest.json").read_text(encoding="utf-8"))
    assert man["files"]["signal_card"] == "signal_card.json"


def test_write_bundle_rejects_non_datetime_index(tmp_path: Path) -> None:
    bad = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1.0]}, index=[0])
    with pytest.raises(TypeError):
        write_quantconnect_bundle(
            tmp_path / "bad",
            symbol="X",
            timeframe="1D",
            start="2024-01-01",
            end="2024-01-02",
            ohlcv_vendor="t",
            df=bad,
        )
