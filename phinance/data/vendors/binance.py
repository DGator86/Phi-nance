"""
phinance.data.vendors.binance
==============================

Binance public kline data vendor — no API key required.

Downloads monthly kline zip files from ``data.binance.vision`` and
assembles them into a continuous normalised OHLCV DataFrame.

Supported timeframes
--------------------
  ``"1D"``, ``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``

Usage
-----
    from phinance.data.vendors.binance import BinanceVendor
    v = BinanceVendor()
    df = v.fetch("BTCUSDT", "1H", "2023-01-01", "2023-06-30")
"""

from __future__ import annotations

import io
import time
import zipfile
from typing import Any, List

import pandas as pd
import requests

from phinance.data.vendors.base import BaseVendor
from phinance.exceptions import DataFetchError, UnsupportedTimeframeError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_BINANCE_BASE = "https://data.binance.vision/data"
_TF_MAP = {"1D": "1d", "1H": "1h", "15m": "15m", "5m": "5m", "1m": "1m"}
_MAX_RETRIES = 3

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]


class BinanceVendor(BaseVendor):
    """Binance public kline data vendor (spot market).

    Parameters
    ----------
    max_retries : int
        Retry count per monthly zip file.
    """

    name = "binance"

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
        """Fetch OHLCV from Binance public data.

        Raises
        ------
        UnsupportedTimeframeError
        DataFetchError
        """
        if timeframe not in _TF_MAP:
            raise UnsupportedTimeframeError(
                f"Binance supports: {', '.join(_TF_MAP)}. Got: {timeframe}"
            )

        interval = _TF_MAP[timeframe]
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if end_dt < start_dt:
            raise DataFetchError("end date must be >= start date")

        months = pd.period_range(
            start=start_dt.to_period("M"),
            end=end_dt.to_period("M"),
            freq="M",
        )

        frames: List[pd.DataFrame] = []
        for m in months:
            df_month = self._fetch_month(symbol, interval, m)
            if df_month is not None:
                frames.append(df_month)

        if not frames:
            raise DataFetchError(
                f"No Binance data found for {symbol} {interval} in range"
            )

        all_df = pd.concat(frames, ignore_index=True)
        all_df["open_time"] = (
            pd.to_datetime(all_df["open_time"], unit="ms", utc=True)
            .dt.tz_localize(None)
        )
        all_df = all_df.set_index("open_time")
        all_df = all_df[
            (all_df.index >= start_dt)
            & (all_df.index <= end_dt + pd.Timedelta(days=1))
        ]
        result = self._normalize(all_df)
        self._sanity_check(result, symbol)
        logger.info(
            "binance: %d rows for %s %s [%s → %s]",
            len(result), symbol, interval, start, end,
        )
        return result

    def _fetch_month(
        self, symbol: str, interval: str, period: Any
    ) -> pd.DataFrame | None:
        """Download and parse one monthly kline zip.  Returns None on 404."""
        ym = f"{period.year:04d}-{period.month:02d}"
        sym_up = symbol.upper()
        url = (
            f"{_BINANCE_BASE}/spot/monthly/klines/"
            f"{sym_up}/{interval}/{sym_up}-{interval}-{ym}.zip"
        )
        last_exc: Exception = DataFetchError("Binance fetch failed")
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("Binance: 404 for %s %s %s", symbol, interval, ym)
                    return None
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=_KLINE_COLS)
                logger.debug("Binance: fetched %s %s %s", symbol, interval, ym)
                return df
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Binance attempt %d/%d failed for %s %s %s: %s. Retry in %ds.",
                    attempt + 1, self.max_retries, symbol, interval, ym, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        logger.warning(
            "Binance: all retries exhausted for %s %s %s", symbol, interval, ym
        )
        return None
