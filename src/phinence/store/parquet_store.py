"""
Parquet store for 1m/5m bars. Layout: data/bars/{ticker}/{year}.parquet.

Load continuous RTH sequences; sanity check "no-gap >5 bars".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phinence.store.schemas import BAR_1M_SCHEMA


class ParquetBarStore:
    """Read/write 1m bars by ticker and year. Same schema for live and historical."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, year: int) -> Path:
        ticker_upper = ticker.upper()
        self.root.joinpath(ticker_upper).mkdir(parents=True, exist_ok=True)
        return self.root / ticker_upper / f"{year}.parquet"

    def write_1m_bars(self, ticker: str, year: int, df: pd.DataFrame) -> None:
        """Write 1m bars; df must have columns matching BAR_1M_SCHEMA."""
        path = self._path(ticker, year)
        table = pa.Table.from_pandas(df, schema=BAR_1M_SCHEMA, preserve_index=False)
        pq.write_table(table, path)

    def read_1m_bars(
        self, ticker: str, year: int | None = None
    ) -> pd.DataFrame:
        """Read 1m bars for ticker (single year or all years in dir)."""
        ticker_upper = ticker.upper()
        base = self.root / ticker_upper
        if not base.exists():
            return pd.DataFrame()
        if year is not None:
            path = base / f"{year}.parquet"
            if not path.exists():
                return pd.DataFrame()
            return pq.read_table(path).to_pandas()
        dfs: list[pd.DataFrame] = []
        for p in sorted(base.glob("*.parquet")):
            dfs.append(pq.read_table(p).to_pandas())
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def list_tickers(self) -> list[str]:
        """List tickers that have at least one parquet file."""
        if not self.root.exists():
            return []
        return [
            d.name for d in self.root.iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        ]

    def stream_rth_1m(
        self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> Iterator[pd.DataFrame]:
        """Yield contiguous RTH 1m chunks; caller can run no-gap >5 bars check."""
        df = self.read_1m_bars(ticker)
        if df.empty:
            return
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if df.empty:
            return
        # Simple chunk by day for "no-gap" checks
        df["date"] = df["timestamp"].dt.date
        for _, grp in df.groupby("date"):
            yield grp.drop(columns=["date"])


def check_no_gap_more_than_n_bars(df: pd.DataFrame, max_gap: int = 5) -> bool:
    """Sanity check: no gap > max_gap minutes in timestamp index."""
    if df.empty or len(df) <= 1:
        return True
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    diff_min = ts.diff().dt.total_seconds() / 60
    return (diff_min.fillna(0) <= (max_gap + 1)).all()
