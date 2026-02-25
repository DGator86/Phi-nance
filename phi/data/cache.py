"""
phi.data.cache — Dataset Cache Manager
=======================================
Fetch once, store locally, reuse forever.

Storage layout:
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}.parquet
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}_metadata.json
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CACHE_ROOT = Path(__file__).parents[2] / "data_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_dir(vendor: str, symbol: str, timeframe: str) -> Path:
    return CACHE_ROOT / vendor / symbol.upper() / timeframe


def _stem(start, end) -> str:
    s = str(start)[:10].replace("-", "")
    e = str(end)[:10].replace("-", "")
    return f"{s}_{e}"


def get_cache_path(vendor: str, symbol: str, timeframe: str, start, end) -> Path:
    return _cache_dir(vendor, symbol, timeframe) / f"{_stem(start, end)}.parquet"


def get_meta_path(vendor: str, symbol: str, timeframe: str, start, end) -> Path:
    return _cache_dir(vendor, symbol, timeframe) / f"{_stem(start, end)}_metadata.json"


def dataset_id(vendor: str, symbol: str, timeframe: str, start, end) -> str:
    key = f"{vendor}_{symbol.upper()}_{timeframe}_{str(start)[:10]}_{str(end)[:10]}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def is_cached(vendor: str, symbol: str, timeframe: str, start, end) -> bool:
    return get_cache_path(vendor, symbol, timeframe, start, end).exists()


def load_dataset(vendor: str, symbol: str, timeframe: str, start, end) -> Optional[pd.DataFrame]:
    """Load a cached dataset.  Returns None if not found or corrupt."""
    p = get_cache_path(vendor, symbol, timeframe, start, end)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def save_dataset(
    df: pd.DataFrame,
    vendor: str,
    symbol: str,
    timeframe: str,
    start,
    end,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a dataset to parquet + write metadata JSON.  Returns parquet path."""
    p = get_cache_path(vendor, symbol, timeframe, start, end)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)

    meta: Dict[str, Any] = {
        "dataset_id":  dataset_id(vendor, symbol, timeframe, start, end),
        "vendor":      vendor,
        "symbol":      symbol.upper(),
        "timeframe":   timeframe,
        "start":       str(start)[:10],
        "end":         str(end)[:10],
        "rows":        len(df),
        "columns":     list(df.columns),
        "cached_at":   datetime.utcnow().isoformat() + "Z",
        "parquet_path": str(p),
    }
    if df.index is not None and len(df) > 0:
        meta["first_bar"] = str(df.index[0])
        meta["last_bar"]  = str(df.index[-1])
    if extra_meta:
        meta.update(extra_meta)

    mp = get_meta_path(vendor, symbol, timeframe, start, end)
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)

    return p


def load_metadata(vendor: str, symbol: str, timeframe: str, start, end) -> Optional[Dict[str, Any]]:
    mp = get_meta_path(vendor, symbol, timeframe, start, end)
    if not mp.exists():
        return None
    try:
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


def list_cached_datasets() -> List[Dict[str, Any]]:
    """Return metadata dicts for all cached datasets, newest first."""
    results = []
    if not CACHE_ROOT.exists():
        return results
    for meta_file in sorted(CACHE_ROOT.rglob("*_metadata.json")):
        try:
            with open(meta_file) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return sorted(results, key=lambda x: x.get("cached_at", ""), reverse=True)


def delete_dataset(vendor: str, symbol: str, timeframe: str, start, end) -> bool:
    """Delete a cached dataset and its metadata.  Returns True if anything deleted."""
    removed = False
    for p in [
        get_cache_path(vendor, symbol, timeframe, start, end),
        get_meta_path(vendor, symbol, timeframe, start, end),
    ]:
        if p.exists():
            p.unlink()
            removed = True
    return removed


def clear_all_cache() -> int:
    """Delete all cached datasets.  Returns number of parquet files removed."""
    count = 0
    if not CACHE_ROOT.exists():
        return 0
    for p in CACHE_ROOT.rglob("*.parquet"):
        p.unlink()
        count += 1
    for p in CACHE_ROOT.rglob("*_metadata.json"):
        p.unlink()
    return count
