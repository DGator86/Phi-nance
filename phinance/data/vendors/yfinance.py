"""
phinance.data.vendors.yfinance
==============================

yfinance vendor adapter — daily and intraday OHLCV, no API key required.

Supported timeframes
--------------------
  Daily   : ``"1D"``                       (unlimited history)
  Intraday: ``"1m"``, ``"5m"``, ``"15m"``, ``"30m"``, ``"1H"``
    - 1m  → last 7 calendar days
    - 5m  → last 60 calendar days
    - 15m → last 60 calendar days
    - 30m → last 60 calendar days
    - 1H  → last 730 calendar days

Usage
-----
    from phinance.data.vendors.yfinance import YFinanceVendor
    v = YFinanceVendor()
    df = v.fetch("SPY", "1D", "2022-01-01", "2023-12-31")
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from phinance.data.vendors.base import BaseVendor
from phinance.exceptions import DataFetchError, UnsupportedTimeframeError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

# yfinance intraday interval map + lookback caps (calendar days)
_YF_INTRADAY_MAP = {
    "1m":  ("1m",  7),
    "5m":  ("5m",  60),
    "15m": ("15m", 60),
    "30m": ("30m", 60),
    "1H":  ("1h",  730),
}

_MAX_RETRIES = 3


class YFinanceVendor(BaseVendor):
    """yfinance data vendor (no API key, no rate limiting for daily data).

    Parameters
    ----------
    max_retries : int
        Number of fetch attempts with exponential backoff.
    """

    name = "yfinance"

    def __init__(self, max_retries: int = _MAX_RETRIES) -> None:
        self.max_retries = max_retries

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV from yfinance with automatic retry.

        Parameters
        ----------
        symbol : str
        timeframe : str — ``"1D"`` or intraday key
        start, end : str — ``"YYYY-MM-DD"``

        Returns
        -------
        pd.DataFrame — normalised OHLCV

        Raises
        ------
        UnsupportedTimeframeError
            For unsupported intraday timeframes.
        DataFetchError
            When all retries are exhausted.
        """
        if timeframe == "1D":
            return self._fetch_daily(symbol, start, end)
        if timeframe in _YF_INTRADAY_MAP:
            return self._fetch_intraday(symbol, timeframe, start, end)
        raise UnsupportedTimeframeError(
            f"yfinance supports: 1D, {', '.join(_YF_INTRADAY_MAP)}. Got: {timeframe}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        last_exc: Exception = DataFetchError(f"No daily data for {symbol}")
        for attempt in range(self.max_retries):
            try:
                tkr = yf.Ticker(symbol)
                df = tkr.history(start=start, end=end, auto_adjust=True)
                if df.empty:
                    raise DataFetchError(f"No daily data for {symbol}")
                result = self._normalize(df)
                self._sanity_check(result, symbol)
                logger.info(
                    "yfinance daily: %d rows for %s [%s → %s]",
                    len(result), symbol, start, end,
                )
                return result
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "yfinance daily attempt %d/%d failed for %s: %s. Retry in %ds.",
                    attempt + 1, self.max_retries, symbol, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise DataFetchError(str(last_exc)) from last_exc

    def _fetch_intraday(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        import yfinance as yf

        interval, max_days = _YF_INTRADAY_MAP[timeframe]
        # Clamp start to lookback cap
        cap_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=max_days)
        if pd.Timestamp(start) < cap_start:
            logger.warning(
                "yfinance %s keeps only %d days. Clamping start %s → %s.",
                interval, max_days, start, cap_start.date(),
            )
            start = str(cap_start.date())

        last_exc: Exception = DataFetchError(
            f"No intraday data for {symbol} {interval}"
        )
        for attempt in range(self.max_retries):
            try:
                tkr = yf.Ticker(symbol)
                df = tkr.history(
                    start=start, end=end, interval=interval, auto_adjust=True
                )
                if df.empty:
                    raise DataFetchError(
                        f"No intraday data for {symbol} {interval}"
                    )
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result = self._normalize(df)
                self._sanity_check(result, symbol)
                logger.info(
                    "yfinance intraday: %d rows for %s %s",
                    len(result), symbol, interval,
                )
                return result
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "yfinance intraday attempt %d/%d failed for %s %s: %s. Retry in %ds.",
                    attempt + 1, self.max_retries, symbol, interval, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise DataFetchError(str(last_exc)) from last_exc
