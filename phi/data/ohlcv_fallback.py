"""Canonical OHLCV loading: **Unusual Whales first**, then **yfinance** (Yahoo).

Use this anywhere the app needs price history so vendor policy stays consistent
(easy mode, trading desk, scripts).
"""

from __future__ import annotations

import pandas as pd

from phi.logging import get_logger

from .cache import fetch_and_cache

logger = get_logger(__name__)

OHLCV_VENDOR_FALLBACK: tuple[str, ...] = ("unusual_whales", "yfinance")


def fetch_ohlcv_uw_then_yf(
    symbol: str,
    start: str,
    end: str,
    *,
    timeframe: str = "1D",
) -> tuple[pd.DataFrame, str]:
    """Return ``(ohlcv, vendor_used)``. Raises if both vendors fail.

    Parameters
    ----------
    symbol
        Equity / ETF ticker.
    start, end
        ISO date strings (inclusive range as implemented by cache fetchers).
    timeframe
        e.g. ``1D`` — passed through to ``fetch_and_cache``.
    """
    sym = symbol.strip().upper()
    last_exc: Exception | None = None
    for vendor in OHLCV_VENDOR_FALLBACK:
        try:
            df = fetch_and_cache(vendor, sym, timeframe, start, end)
            if df is not None and not df.empty:
                logger.info("OHLCV for %s: using vendor=%s rows=%s", sym, vendor, len(df))
                return df, vendor
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("OHLCV fetch %s via %s failed: %s", sym, vendor, exc)
    raise RuntimeError(
        f"Could not load OHLCV for {sym} ({start} → {end}). "
        f"Set UNUSUAL_WHALES_API_KEY and/or check network. Last error: {last_exc}"
    )
