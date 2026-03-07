from __future__ import annotations

import time

import pandas as pd

from phinance.data.optimised_cache import OptimisedCache


def test_optimised_cache_set_get_and_ttl_expiry():
    cache = OptimisedCache(max_size_mb=1, default_ttl_seconds=0.05)
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    time.sleep(0.06)
    assert cache.get("k") is None


def test_optimised_cache_lru_eviction():
    cache = OptimisedCache(max_size_mb=0)  # force immediate eviction
    cache.set("a", "x" * 100)
    cache.set("b", "y" * 100)
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.stats()["evictions"] >= 1


def test_optimised_cache_dataframe_size_estimation():
    cache = OptimisedCache(max_size_mb=10)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cache.set("df", df)
    out = cache.get("df")
    assert out is not None
    assert out.equals(df)
