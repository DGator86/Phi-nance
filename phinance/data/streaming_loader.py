"""Streaming loader for chunked training/backtest data access."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from phinance.data.memmap_store import MemmapStore
from phinance.data.prefetcher import Prefetcher


class StreamingDataLoader:
    """Yield mini-batches from numpy arrays or symbol memmaps."""

    def __init__(
        self,
        data: np.ndarray | None = None,
        *,
        store: MemmapStore | None = None,
        symbol: str | None = None,
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
        prefetch: int = 0,
        random_seed: int = 7,
    ) -> None:
        if data is None and (store is None or symbol is None):
            raise ValueError("Provide `data` or (`store` and `symbol`).")
        self._direct_data = None if data is None else np.asarray(data)
        self.store = store
        self.symbol = symbol
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.prefetch = int(prefetch)
        self._rng = np.random.default_rng(random_seed)

    def _dataset(self) -> np.ndarray:
        if self._direct_data is not None:
            return self._direct_data
        assert self.store is not None and self.symbol is not None
        return self.store.load(self.symbol)

    def _iter_batches(self) -> Iterator[np.ndarray]:
        data = self._dataset()
        n = int(data.shape[0])
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and (end - start) < self.batch_size:
                continue
            window = indices[start:end]
            if self.shuffle:
                yield np.asarray(data[window])
            else:
                yield data[start:end]

    def __iter__(self) -> Iterator[np.ndarray]:
        iterator: Iterator[np.ndarray] = self._iter_batches()
        if self.prefetch > 0:
            return iter(Prefetcher(iterator, prefetch=self.prefetch))
        return iterator
