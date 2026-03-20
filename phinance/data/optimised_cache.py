"""Optimised multi-level cache for market datasets and derived windows."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable

import pandas as pd


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class OptimisedCache:
    """Thread-safe in-memory LRU+TTL cache with optional disk fallback.

    Notes
    -----
    * `max_size_mb` bounds total estimated memory for in-memory objects.
    * Entries may be marked with custom `ttl_seconds` per key.
    * When a disk cache backend is provided, misses may load from disk and
      hydrate memory for subsequent requests.
    """

    def __init__(
        self,
        max_size_mb: int = 1024,
        default_ttl_seconds: float | None = None,
        disk_cache: Any | None = None,
    ) -> None:
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl_seconds = default_ttl_seconds
        self.disk_cache = disk_cache

        self._lock = threading.RLock()
        self._entries: OrderedDict[Hashable, tuple[Any, int, float | None]] = OrderedDict()
        self._current_bytes = 0
        self._stats = CacheStats()

    def _estimate_size_bytes(self, value: Any) -> int:
        if isinstance(value, pd.DataFrame):
            return int(value.memory_usage(deep=True, index=True).sum())
        if isinstance(value, pd.Series):
            return int(value.memory_usage(deep=True, index=True))
        if hasattr(value, "nbytes"):
            return int(value.nbytes)
        return max(1, len(repr(value).encode("utf-8")))

    def _is_expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and time.time() > expires_at

    def _purge_if_needed(self) -> None:
        while self._current_bytes > self.max_size_bytes and self._entries:
            _, (_, size_bytes, _) = self._entries.popitem(last=False)
            self._current_bytes -= size_bytes
            self._stats.evictions += 1

    def set(self, key: Hashable, value: Any, ttl_seconds: float | None = None) -> None:
        entry_size = self._estimate_size_bytes(value)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = None if ttl is None else time.time() + float(ttl)

        with self._lock:
            if key in self._entries:
                _, existing_size, _ = self._entries.pop(key)
                self._current_bytes -= existing_size
            self._entries[key] = (value, entry_size, expires_at)
            self._entries.move_to_end(key, last=True)
            self._current_bytes += entry_size
            self._purge_if_needed()

    def get(self, key: Hashable, default: Any | None = None) -> Any | None:
        with self._lock:
            rec = self._entries.get(key)
            if rec is not None:
                value, size_bytes, expires_at = rec
                if self._is_expired(expires_at):
                    self._entries.pop(key, None)
                    self._current_bytes -= size_bytes
                    self._stats.expirations += 1
                else:
                    self._entries.move_to_end(key, last=True)
                    self._stats.hits += 1
                    return value

        disk_value = self._load_from_disk(key)
        if disk_value is not None:
            self.set(key, disk_value)
            self._stats.hits += 1
            return disk_value

        self._stats.misses += 1
        return default

    def delete(self, key: Hashable) -> None:
        with self._lock:
            rec = self._entries.pop(key, None)
            if rec is not None:
                _, size_bytes, _ = rec
                self._current_bytes -= size_bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "expirations": self._stats.expirations,
            "items": len(self._entries),
            "size_bytes": self._current_bytes,
        }

    def _load_from_disk(self, key: Hashable) -> Any | None:
        if self.disk_cache is None or not isinstance(key, tuple) or len(key) < 5:
            return None
        vendor, symbol, timeframe, start, end = key[:5]
        try:
            return self.disk_cache.load(vendor, symbol, timeframe, start, end)
        except Exception:
            return None
