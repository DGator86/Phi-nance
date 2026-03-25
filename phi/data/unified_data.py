"""
Canonical OHLCV access for ecosystem adapters (Lumibot, TensorTrade, notebooks).

All paths resolve through the same parquet cache and vendor routing as the rest
of Phi-nance — use this module when another repo should not reimplement fetch logic.

See ``docs/ecosystem_integration.md`` for Lumibot / TensorTrade / agent-cli wiring.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phi.data.cache import fetch_and_cache, get_cached_dataset
from phi.logging import get_logger

logger = get_logger(__name__)


def get_ohlcv(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1D",
    vendor: str = "yfinance",
    *,
    force_refresh: bool = False,
    fallback_vendors: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Return OHLCV from the Phi-nance data spine (fetch + cache).

    Parameters mirror :func:`~phi.data.cache.fetch_and_cache` intent; argument
    order is optimized for “symbol-first” calls from external frameworks.

    Examples
    --------
    >>> from phi.data.unified_data import get_ohlcv
    >>> df = get_ohlcv("SPY", "2023-01-01", "2024-01-01", "1D", "yfinance")
    """
    return fetch_and_cache(
        vendor,
        symbol,
        timeframe,
        str(start)[:10],
        str(end)[:10],
        force_refresh=force_refresh,
        fallback_vendors=fallback_vendors,
        **kwargs,
    )


def load_ohlcv_cached_first(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1D",
    vendor: str = "yfinance",
) -> pd.DataFrame:
    """Prefer a cold cache read; fetch only on miss (faster for batch RL / exports)."""
    sym = symbol.upper()
    start_s, end_s = str(start)[:10], str(end)[:10]
    cached = get_cached_dataset(vendor, sym, timeframe, start_s, end_s)
    if cached is not None and not cached.empty:
        logger.info("unified_data: cache hit %s %s %s %s-%s", vendor, sym, timeframe, start_s, end_s)
        return cached
    return get_ohlcv(sym, start_s, end_s, timeframe, vendor)


def try_external_ohlcv(
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
    *,
    import_path: str | None = None,
) -> pd.DataFrame | None:
    """
    Optional hook for a sibling checkout (e.g. agent-cli) without hard deps.

    Set ``import_path`` to a ``module:callable`` string. The callable should
    accept ``(symbol, start, end, timeframe)`` and return a DataFrame or None.

    If ``import_path`` is omitted, reads env ``PHINANCE_ECOSYSTEM_OHLCV_HOOK``.
    """
    import importlib
    import os

    spec = import_path or os.environ.get("PHINANCE_ECOSYSTEM_OHLCV_HOOK", "").strip()
    if not spec or ":" not in spec:
        return None
    mod_name, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name.strip())
        fn = getattr(mod, attr.strip())
        out = fn(symbol, str(start)[:10], str(end)[:10], timeframe)
        if out is None or (isinstance(out, pd.DataFrame) and out.empty):
            return None
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"hook {spec} must return DataFrame or None, got {type(out)}")
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("unified_data: external OHLCV hook %r failed: %s", spec, exc)
        return None


def get_ohlcv_with_optional_hook(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1D",
    vendor: str = "yfinance",
    *,
    hook_import_path: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Try Phi-nance spine first; if empty and a hook is configured, try external source.

    Use this when Hyperliquid or another runtime feed should only supplement
    missing cache ranges — not replace vendor-of-record history.
    """
    try:
        df = get_ohlcv(symbol, start, end, timeframe, vendor, **kwargs)
        if df is not None and not df.empty:
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("unified_data: primary fetch failed: %s", exc)

    ext = try_external_ohlcv(symbol, start, end, timeframe, import_path=hook_import_path)
    if ext is not None and not ext.empty:
        return ext

    raise ValueError(f"No OHLCV available for {symbol} {timeframe} {start}..{end}")
