"""
AssignmentEngine — strict router, not a processor.

Loads 1m bars, resamples to 5m, attaches chain snapshot; outputs AssignedPacket
with coverage flags and warnings. Missing data never crashes; coverage drops, confidence → 0.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from phinence.contracts.assigned_packet import AssignedPacket, CoverageFlag
from phinence.store.parquet_store import ParquetBarStore
from phinence.store.memory_store import InMemoryBarStore

# Any store that implements read_1m_bars(ticker) and list_tickers()
BarStore = ParquetBarStore | InMemoryBarStore


def resample_1m_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Derive 5m OHLCV from 1m. Expects columns timestamp, open, high, low, close, volume."""
    if df_1m.empty or len(df_1m) < 5:
        return pd.DataFrame()
    df = df_1m.set_index(pd.to_datetime(df_1m["timestamp"])).sort_index()
    rule = "5min"
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    out["timestamp"] = out.index
    return out.reset_index(drop=True)[["timestamp", "open", "high", "low", "close", "volume"]]


class AssignmentEngine:
    """Build AssignedPacket from store + optional live chain snapshot."""

    def __init__(
        self,
        bar_store: BarStore,
        min_bars_1m: int = 5,
        min_bars_5m: int = 5,
    ) -> None:
        self.bar_store = bar_store
        self.min_bars_1m = min_bars_1m
        self.min_bars_5m = min_bars_5m

    def assign(
        self,
        ticker: str,
        as_of: datetime,
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
        chain_snapshot: dict[str, Any] | None = None,
    ) -> AssignedPacket:
        """
        Load 1m bars for [start_ts, end_ts], resample to 5m, set coverage flags.
        If no range given, load full available history (for backtest).
        """
        warnings: list[str] = []
        if start_ts is None or end_ts is None:
            df_1m = self.bar_store.read_1m_bars(ticker)
        else:
            df_1m = self.bar_store.read_1m_bars(ticker)
            if not df_1m.empty:
                df_1m = df_1m[
                    (pd.to_datetime(df_1m["timestamp"]) >= start_ts)
                    & (pd.to_datetime(df_1m["timestamp"]) <= end_ts)
                ]
        bars_1m = df_1m.to_dict("records") if not df_1m.empty else []
        for r in bars_1m:
            if "timestamp" in r and hasattr(r["timestamp"], "isoformat"):
                r["timestamp"] = r["timestamp"].isoformat()
        if len(bars_1m) < self.min_bars_1m:
            coverage_1m = CoverageFlag.MISSING if len(bars_1m) == 0 else CoverageFlag.PARTIAL
            warnings.append(f"1m bars below minimum ({len(bars_1m)} < {self.min_bars_1m})")
        else:
            coverage_1m = CoverageFlag.FULL

        df_5m = resample_1m_to_5m(df_1m) if not df_1m.empty else pd.DataFrame()
        bars_5m = df_5m.to_dict("records") if not df_5m.empty else []
        for r in bars_5m:
            if "timestamp" in r and hasattr(r["timestamp"], "isoformat"):
                r["timestamp"] = r["timestamp"].isoformat()
        if len(bars_5m) < self.min_bars_5m:
            coverage_5m = CoverageFlag.MISSING if len(bars_5m) == 0 else CoverageFlag.PARTIAL
        else:
            coverage_5m = CoverageFlag.FULL

        chain_cov = CoverageFlag.FULL if chain_snapshot else CoverageFlag.MISSING
        return AssignedPacket(
            ticker=ticker,
            as_of=as_of,
            bars_1m=bars_1m,
            bars_5m=bars_5m,
            coverage_1m=coverage_1m,
            coverage_5m=coverage_5m,
            chain_snapshot=chain_snapshot,
            chain_coverage=chain_cov,
            warnings=warnings,
        )
