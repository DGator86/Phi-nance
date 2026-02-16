"""Protocol for bar stores so AssignmentEngine can use Parquet or in-memory data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


class BarStoreProtocol(Protocol):
    """Minimal interface for 1m bar access. Implemented by ParquetBarStore and InMemoryBarStore."""

    def read_1m_bars(self, ticker: str, year: int | None = None) -> "pd.DataFrame":
        ...
    def list_tickers(self) -> list[str]:
        ...
