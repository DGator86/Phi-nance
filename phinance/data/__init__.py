"""
phinance.data — Data acquisition, caching, and utilities.

Public API
----------
    from phinance.data import fetch_and_cache, DataCache, list_cached_datasets

Sub-modules
-----------
  cache      — Parquet-based dataset cache (DataCache + fetch_and_cache)
  utils      — Data cleaning, resampling, and normalisation helpers
  vendors/   — Pluggable data-source adapters
"""

from phinance.data.cache import (
    DataCache,
    fetch_and_cache,
    get_cached_dataset,
    list_cached_datasets,
)
from phinance.data.memmap_store import MemmapStore
from phinance.data.optimised_cache import OptimisedCache
from phinance.data.prefetcher import Prefetcher
from phinance.data.streaming_loader import StreamingDataLoader

__all__ = [
    "DataCache",
    "fetch_and_cache",
    "get_cached_dataset",
    "list_cached_datasets",
    "OptimisedCache",
    "MemmapStore",
    "StreamingDataLoader",
    "Prefetcher",
]
