"""
phinance.data.vendors.alphavantage
===================================

Alpha Vantage vendor adapter.

For daily (``"1D"``) timeframe this delegates to YFinanceVendor since
yfinance provides the same data without rate-limiting concerns.

For intraday timeframes (``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``) it uses
the Alpha Vantage TIME_SERIES_INTRADAY endpoint via
``regime_engine.data_fetcher.AlphaVantageFetcher``.

Usage
-----
    from phinance.data.vendors.alphavantage import AlphaVantageVendor
    v = AlphaVantageVendor(api_key="YOUR_KEY")
    df = v.fetch("AAPL", "15m", "2024-01-01", "2024-03-31")
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from phinance.data.vendors.base import BaseVendor
from phinance.exceptions import DataFetchError, UnsupportedTimeframeError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_INTERVAL_MAP = {
    "4H": "60min",
    "1H": "60min",
    "15m": "15min",
    "5m": "5min",
    "1m": "1min",
}

_MAX_RETRIES = 3


class AlphaVantageVendor(BaseVendor):
    """Alpha Vantage OHLCV vendor.

    Parameters
    ----------
    api_key : str, optional
        Alpha Vantage API key.  Falls back to ``AV_API_KEY`` environment var.
    max_retries : int
        Retry count with exponential backoff.
    """

    name = "alphavantage"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        import os

        self.api_key: Optional[str] = api_key or os.environ.get("AV_API_KEY")
        self.max_retries = max_retries

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Alpha Vantage.

        Delegates daily data to ``YFinanceVendor`` to avoid AV rate limits.

        Raises
        ------
        UnsupportedTimeframeError
        DataFetchError
        """
        if timeframe == "1D":
            from phinance.data.vendors.yfinance import YFinanceVendor
            return YFinanceVendor(max_retries=self.max_retries).fetch(
                symbol, "1D", start, end
            )

        if timeframe not in _INTERVAL_MAP:
            raise UnsupportedTimeframeError(
                f"Alpha Vantage supports: 1D, {', '.join(_INTERVAL_MAP)}. Got: {timeframe}"
            )

        return self._fetch_intraday(symbol, timeframe, start, end)

    def _fetch_intraday(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        last_exc: Exception = DataFetchError(f"No AV data for {symbol}")
        interval = _INTERVAL_MAP[timeframe]

        for attempt in range(self.max_retries):
            try:
                from regime_engine.data_fetcher import AlphaVantageFetcher

                av = AlphaVantageFetcher(api_key=self.api_key)
                raw = av.intraday(
                    symbol, interval=interval, outputsize="full", cache_ttl=0
                )
                if raw.empty:
                    raise DataFetchError(f"No AV data for {symbol}")
                windowed = raw[
                    (raw.index >= pd.Timestamp(start))
                    & (raw.index <= pd.Timestamp(end))
                ]
                result = self._normalize(windowed)
                self._sanity_check(result, symbol)
                logger.info(
                    "alphavantage: %d rows for %s %s", len(result), symbol, interval
                )
                return result
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "alphavantage attempt %d/%d failed for %s: %s. Retry in %ds.",
                    attempt + 1, self.max_retries, symbol, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        raise DataFetchError(str(last_exc)) from last_exc
