"""In-memory bar store for backtests and tests (no Polygon data required)."""

from __future__ import annotations

from typing import Any

import pandas as pd


class InMemoryBarStore:
    """Holds 1m bars per ticker in memory. read_1m_bars(ticker) returns full history."""

    def __init__(self) -> None:
        self._data: dict[str, pd.DataFrame] = {}

    def put_1m_bars(self, ticker: str, df: pd.DataFrame) -> None:
        """Set 1m bars for ticker. df must have timestamp, open, high, low, close, volume."""
        ticker = ticker.upper()
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._data[ticker] = df

    def read_1m_bars(self, ticker: str, year: int | None = None) -> pd.DataFrame:
        """Return 1m bars for ticker, optionally filtered by year."""
        df = self._data.get(ticker.upper(), pd.DataFrame())
        if df.empty:
            return df
        if year is not None:
            ts = pd.to_datetime(df["timestamp"])
            df = df[(ts.dt.year == year)]
        return df.copy()

    def list_tickers(self) -> list[str]:
        return list(self._data.keys())
