"""Persistent cache helpers for live-data API responses."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class PersistentCache:
    """Simple SQLite-backed key/value cache with expiry timestamps."""

    def __init__(self, db_path: str | Path = ".cache/live_data_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at)"
        )
        self._conn.commit()

    def get(self, cache_key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None

        value_json, expires_at = row
        now = time.time()
        if expires_at is not None and expires_at < now:
            self.delete(cache_key)
            return None
        return json.loads(value_json)

    def set(self, cache_key: str, value: Any, expiry_seconds: float | None = None) -> None:
        now = time.time()
        expires_at = None if expiry_seconds is None else now + float(expiry_seconds)
        payload = json.dumps(value, default=str)
        self._conn.execute(
            """
            INSERT INTO cache_entries(cache_key, value_json, created_at, expires_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                value_json=excluded.value_json,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at
            """,
            (cache_key, payload, now, expires_at),
        )
        self._conn.commit()

    def delete(self, cache_key: str) -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE cache_key=?", (cache_key,))
        self._conn.commit()

    def purge_expired(self) -> int:
        now = time.time()
        cur = self._conn.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
