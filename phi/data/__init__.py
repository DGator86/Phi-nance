"""
Phi-nance Data Module — Fetch and cache historical datasets.

Usage:
    from phi.data import DataCache, fetch_and_cache
    df = fetch_and_cache("alphavantage", "SPY", "1D", "2020-01-01", "2024-12-31")
"""

from phi.exceptions import CacheCorruptedError, DataFetchError
from phi.logging import get_logger, setup_logging

from .cache import (
    DataCache,
    auto_fetch_and_cache,
    fetch_and_cache,
    get_cached_dataset,
    is_cache_stale,
    list_cached_datasets,
)
from .ohlcv_fallback import OHLCV_VENDOR_FALLBACK, fetch_ohlcv_uw_then_yf
from .unified_data import (
    get_ohlcv,
    get_ohlcv_with_optional_hook,
    load_ohlcv_cached_first,
    try_external_ohlcv,
)

from .vendor_postgres import PostgresOptionsVendor

setup_logging("phi")
logger = get_logger(__name__)

__all__ = [
    "DataCache",
    "CacheCorruptedError",
    "DataFetchError",
    "fetch_and_cache",
    "auto_fetch_and_cache",
    "get_cached_dataset",
    "get_ohlcv",
    "get_ohlcv_with_optional_hook",
    "load_ohlcv_cached_first",
    "try_external_ohlcv",
    "is_cache_stale",
    "list_cached_datasets",
    "OHLCV_VENDOR_FALLBACK",
    "fetch_ohlcv_uw_then_yf",
    "PostgresOptionsVendor",
    "logger",
]
