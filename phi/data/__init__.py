"""
Phi-nance Data Module — Fetch and cache historical datasets.

Usage:
    from phi.data import DataCache, fetch_and_cache
    df = fetch_and_cache("alphavantage", "SPY", "1D", "2020-01-01", "2024-12-31")
"""

from .cache import DataCache, fetch_and_cache, get_cached_dataset, list_cached_datasets

__all__ = [
    "DataCache",
    "fetch_and_cache",
    "get_cached_dataset",
    "list_cached_datasets",
]
