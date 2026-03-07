"""Asynchronous prefetching utilities for iterable data pipelines."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
_SENTINEL = object()


class Prefetcher(Iterator[T]):
    """Background-thread prefetch wrapper around a source iterable."""

    def __init__(self, iterable: Iterable[T], prefetch: int = 2) -> None:
        self._iterable = iter(iterable)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, int(prefetch)))
        self._exc: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._iterable:
                self._queue.put(item)
        except Exception as exc:  # noqa: BLE001
            self._exc = exc
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self) -> "Prefetcher[T]":
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _SENTINEL:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item  # type: ignore[return-value]
