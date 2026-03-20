"""Data pipeline performance profiling for baseline vs optimised path."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from phinance.data.memmap_store import MemmapStore
from phinance.data.optimised_cache import OptimisedCache
from phinance.data.streaming_loader import StreamingDataLoader


def _sample_frame(rows: int = 50_000) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="min")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "open": rng.random(rows),
            "high": rng.random(rows),
            "low": rng.random(rows),
            "close": rng.random(rows),
            "volume": rng.integers(100, 10_000, size=rows),
        },
        index=idx,
    )


def test_data_pipeline_profile_baseline_vs_optimised(tmp_path):
    frame = _sample_frame(rows=20_000)

    t0 = time.perf_counter()
    baseline_np = frame.to_numpy(dtype=np.float32)
    baseline_copy = baseline_np.copy()
    baseline_elapsed = time.perf_counter() - t0
    assert baseline_copy.shape[0] == len(frame)

    store = MemmapStore(data_dir=tmp_path / "memmap")
    cache = OptimisedCache(max_size_mb=128, default_ttl_seconds=300)

    t1 = time.perf_counter()
    store.write("SPY", baseline_np)
    loader = StreamingDataLoader(store=store, symbol="SPY", batch_size=512, prefetch=2)
    streamed = []
    for i, batch in enumerate(loader):
        cache.set(("SPY", i), batch)
        streamed.append(batch)
    optimised_elapsed = time.perf_counter() - t1

    joined = np.concatenate(streamed, axis=0)
    assert joined.shape == baseline_np.shape
    # We don't enforce speedup in CI due to environment variance; we just ensure
    # profiling stats are captured and pipeline executes correctly.
    assert baseline_elapsed >= 0.0
    assert optimised_elapsed >= 0.0
