"""LOB event interfaces and data loaders."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class LobEvent:
    """Standardized limit order book event."""

    timestamp: pd.Timestamp
    event_type: str
    price: float
    volume: float
    side: str
    order_id: str | None = None


_REQUIRED_COLUMNS = ("timestamp", "event_type", "price", "volume", "side")


def _normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["event_type"] = out["event_type"].astype(str).str.lower()
    out["side"] = out["side"].astype(str).str.lower()
    out["price"] = out["price"].astype(float)
    out["volume"] = out["volume"].astype(float)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def load_lobster_csv(path: str | Path) -> pd.DataFrame:
    """Load LOBSTER-style events into the standard schema."""
    df = pd.read_csv(path)
    mapped = pd.DataFrame(
        {
            "timestamp": df.get("time", df.get("timestamp")),
            "event_type": df["type"].map(
                {
                    1: "add",
                    2: "cancel",
                    3: "delete",
                    4: "trade",
                    5: "trade",
                }
            ).fillna("add"),
            "order_id": df.get("order_id"),
            "volume": df.get("size", df.get("volume", 0.0)),
            "price": df["price"].astype(float),
            "side": df.get("direction", 1).map({1: "buy", -1: "sell"}).fillna("buy"),
        }
    )
    return _normalize_events(mapped)


def load_dukascopy_ticks(path: str | Path) -> pd.DataFrame:
    """Load Dukascopy bid/ask tick data into quote update events."""
    df = pd.read_csv(path)
    ts = df.get("timestamp", df.get("time"))
    bid_events = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": "quote",
            "price": df["bid"],
            "volume": df.get("bid_volume", 1.0),
            "side": "buy",
        }
    )
    ask_events = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": "quote",
            "price": df["ask"],
            "volume": df.get("ask_volume", 1.0),
            "side": "sell",
        }
    )
    return _normalize_events(pd.concat([bid_events, ask_events], ignore_index=True))


def load_custom_csv(path: str | Path, column_map: dict[str, str]) -> pd.DataFrame:
    """Load user CSV by mapping custom columns to the standard schema."""
    df = pd.read_csv(path)
    remapped = {
        target: df[source]
        for target, source in column_map.items()
        if source in df.columns
    }
    standardized = pd.DataFrame(remapped)
    if "order_id" not in standardized.columns:
        standardized["order_id"] = None
    return _normalize_events(standardized)


def iter_events(df: pd.DataFrame, batch_size: int | None = None) -> Iterator[LobEvent | list[LobEvent]]:
    """Yield standardized events one-by-one or in fixed-size batches."""
    events = [
        LobEvent(
            timestamp=row.timestamp,
            event_type=row.event_type,
            price=float(row.price),
            volume=float(row.volume),
            side=str(row.side),
            order_id=None if pd.isna(getattr(row, "order_id", None)) else str(row.order_id),
        )
        for row in _normalize_events(df).itertuples(index=False)
    ]

    if batch_size is None or batch_size <= 0:
        for event in events:
            yield event
        return

    for index in range(0, len(events), batch_size):
        yield events[index : index + batch_size]


def ensure_event_iterator(source: Iterable[LobEvent] | pd.DataFrame) -> Iterator[LobEvent]:
    """Convert event source types into a plain event iterator."""
    if isinstance(source, pd.DataFrame):
        for event in iter_events(source):
            if isinstance(event, list):
                for item in event:
                    yield item
            else:
                yield event
        return

    for event in source:
        yield event
