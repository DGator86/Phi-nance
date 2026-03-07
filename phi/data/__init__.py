"""
Phi-nance Data Module — Fetch and cache historical datasets.

Usage:
    from phi.data import DataCache, fetch_and_cache
    df = fetch_and_cache("alphavantage", "SPY", "1D", "2020-01-01", "2024-12-31")
"""

from phi.logging import get_logger, setup_logging

from .cache import (
    DataCache,
    CacheCorruptedError,
    DataFetchError,
    auto_fetch_and_cache,
    fetch_and_cache,
    get_cached_dataset,
    is_cache_stale,
    list_cached_datasets,
)

setup_logging("phi")
logger = get_logger(__name__)

__all__ = [
    "DataCache",
    "CacheCorruptedError",
    "DataFetchError",
    "fetch_and_cache",
    "auto_fetch_and_cache",
    "get_cached_dataset",
    "is_cache_stale",
    "list_cached_datasets",
    "logger",
]
