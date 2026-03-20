from __future__ import annotations

import numpy as np

from phinance.data.memmap_store import MemmapStore
from phinance.data.streaming_loader import StreamingDataLoader


def test_streaming_loader_batches_from_numpy():
    data = np.arange(25, dtype=np.float32).reshape(25, 1)
    loader = StreamingDataLoader(data=data, batch_size=8, shuffle=False, prefetch=2)

    batches = list(loader)
    assert [b.shape[0] for b in batches] == [8, 8, 8, 1]
    np.testing.assert_allclose(np.concatenate(batches, axis=0), data)


def test_streaming_loader_batches_from_memmap(tmp_path):
    store = MemmapStore(data_dir=tmp_path)
    data = np.arange(40, dtype=np.float32).reshape(20, 2)
    store.write("QQQ", data)

    loader = StreamingDataLoader(store=store, symbol="QQQ", batch_size=6, shuffle=False)
    out = np.concatenate(list(loader), axis=0)
    np.testing.assert_allclose(out, data)
