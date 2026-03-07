from __future__ import annotations

import numpy as np

from phinance.data.memmap_store import MemmapStore


def test_memmap_store_write_load_and_window(tmp_path):
    store = MemmapStore(data_dir=tmp_path)
    arr = np.arange(60, dtype=np.float32).reshape(20, 3)

    path = store.write("SPY", arr)
    assert path.exists()
    assert store.exists("SPY")

    loaded = store.load("SPY")
    assert loaded.shape == arr.shape
    np.testing.assert_allclose(np.asarray(loaded), arr)

    window = store.get_window("SPY", 5, 10)
    np.testing.assert_allclose(np.asarray(window), arr[5:10])
