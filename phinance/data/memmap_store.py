"""Memory-mapped storage for large symbol feature/price arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MemmapStore:
    """Persist and load symbol arrays through ``numpy.memmap``.

    Data is stored in per-symbol ``.dat`` files and lightweight metadata JSON.
    """

    def __init__(self, data_dir: str | Path = "data/memmap") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_file(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.dat"

    def _meta_file(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.meta.json"

    def write(self, symbol: str, array: np.ndarray, dtype: np.dtype | str = np.float32) -> Path:
        arr = np.asarray(array, dtype=dtype)
        file_path = self._symbol_file(symbol)
        mm = np.memmap(file_path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()

        meta = {
            "symbol": symbol.upper(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "path": str(file_path),
        }
        with self._meta_file(symbol).open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return file_path

    def load(self, symbol: str, mode: str = "r") -> np.memmap:
        meta = self.metadata(symbol)
        return np.memmap(
            self._symbol_file(symbol),
            mode=mode,
            dtype=np.dtype(meta["dtype"]),
            shape=tuple(meta["shape"]),
        )

    def metadata(self, symbol: str) -> dict:
        meta_path = self._meta_file(symbol)
        with meta_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def exists(self, symbol: str) -> bool:
        return self._symbol_file(symbol).exists() and self._meta_file(symbol).exists()

    def get_window(self, symbol: str, start: int, end: int, mode: str = "r") -> np.ndarray:
        mm = self.load(symbol, mode=mode)
        return mm[start:end]
