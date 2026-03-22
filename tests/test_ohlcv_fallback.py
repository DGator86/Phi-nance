"""phi.data.ohlcv_fallback — UW then yfinance."""

from __future__ import annotations

import pandas as pd
import pytest

import phi.data.ohlcv_fallback as fb


def test_fetch_ohlcv_uw_then_yf_first_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch(vendor: str, sym: str, tf: str, start: str, end: str) -> pd.DataFrame:
        calls.append(vendor)
        if vendor == "unusual_whales":
            return pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
        return pd.DataFrame()

    monkeypatch.setattr(fb, "fetch_and_cache", fake_fetch)
    df, v = fb.fetch_ohlcv_uw_then_yf("SPY", "2024-01-01", "2024-01-31")
    assert v == "unusual_whales"
    assert len(df) == 1
    assert calls == ["unusual_whales"]


def test_fetch_ohlcv_uw_then_yf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(vendor: str, *_a, **_kw) -> pd.DataFrame:
        if vendor == "unusual_whales":
            raise RuntimeError("no key")
        return pd.DataFrame({"close": [2.0]}, index=pd.date_range("2024-01-01", periods=1))

    monkeypatch.setattr(fb, "fetch_and_cache", fake_fetch)
    df, v = fb.fetch_ohlcv_uw_then_yf("QQQ", "2024-01-01", "2024-01-31")
    assert v == "yfinance"
    assert float(df["close"].iloc[-1]) == 2.0
