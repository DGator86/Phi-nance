"""Data-stream utilities for polling live data sources."""

from __future__ import annotations

import time
from typing import Any


class PollingDataStream:
    def __init__(self, manager: Any, data_type: str, interval_seconds: float = 1.0) -> None:
        self.manager = manager
        self.data_type = data_type
        self.interval_seconds = interval_seconds
        self._running = False

    def run(self, callback, **fetch_kwargs: Any) -> None:
        self._running = True
        while self._running:
            payload = self.manager.fetch(self.data_type, **fetch_kwargs)
            callback(payload)
            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False
