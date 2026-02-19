"""
Alpha Vantage Data Fetcher
===========================
Fetches 1-minute OHLCV from Alpha Vantage and returns DataFrames
compatible with the regime engine.

API key:   PLN25H3ESMM1IRBN
MCP token: G7jhPGMv69WcDYerwJ6VZ2Tw6upJ  (mcp.alphavantage.co)

Rate limits
-----------
Free tier  : 25 requests/day, 5 requests/minute
Premium    : varies by plan

Key endpoints used
------------------
TIME_SERIES_INTRADAY          — 1m/5m/15m/30m/60m OHLCV
  outputsize=compact          → last 100 bars (latest trading session)
  outputsize=full             → up to 30 days
  month=YYYY-MM               → specific month of historical 1m data

GLOBAL_QUOTE                  — latest price snapshot (for live scoring)
SYMBOL_SEARCH                 — ticker lookup

Usage
-----
>>> from regime_engine.data_fetcher import AlphaVantageFetcher
>>> av = AlphaVantageFetcher()   # reads key from env or config
>>> df = av.intraday('AAPL', interval='1min')
>>> df = av.intraday_month('AAPL', month='2025-01')
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import pathlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AV_BASE_URL    = "https://www.alphavantage.co/query"
AV_API_KEY     = "PLN25H3ESMM1IRBN"
AV_MCP_TOKEN   = "G7jhPGMv69WcDYerwJ6VZ2Tw6upJ"   # mcp.alphavantage.co bearer token
AV_MCP_URL     = "https://mcp.alphavantage.co/"

_OHLCV_RENAME = {
    "1. open":   "open",
    "2. high":   "high",
    "3. low":    "low",
    "4. close":  "close",
    "5. volume": "volume",
}

_INTERVAL_KEY = {
    "1min":  "Time Series (1min)",
    "5min":  "Time Series (5min)",
    "15min": "Time Series (15min)",
    "30min": "Time Series (30min)",
    "60min": "Time Series (60min)",
}


# ──────────────────────────────────────────────────────────────────────────────
# Local file cache
# ──────────────────────────────────────────────────────────────────────────────

class _FileCache:
    """Simple JSON file cache for API responses.  Avoids repeat calls."""

    def __init__(self, cache_dir: str | pathlib.Path = ".av_cache") -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, params: Dict) -> str:
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def get(self, params: Dict, ttl_minutes: int = 5) -> Optional[Dict]:
        """Return cached value if fresh, else None."""
        path = self.cache_dir / f"{self._key(params)}.json"
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(minutes=ttl_minutes):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, params: Dict, data: Dict) -> None:
        path = self.cache_dir / f"{self._key(params)}.json"
        with open(path, "w") as f:
            json.dump(data, f)


# ──────────────────────────────────────────────────────────────────────────────
# Fetcher
# ──────────────────────────────────────────────────────────────────────────────

class AlphaVantageFetcher:
    """
    Alpha Vantage REST client that returns regime-engine-ready DataFrames.

    Parameters
    ----------
    api_key    : str — Alpha Vantage API key
                 Defaults to env var AV_API_KEY, then the built-in key.
    cache_dir  : path for local JSON cache (default: .av_cache/)
    rate_limit : min seconds between API calls (default 12 → 5 req/min)
    timeout    : HTTP request timeout in seconds

    Examples
    --------
    >>> av = AlphaVantageFetcher()
    >>> df_1m = av.intraday('AAPL', interval='1min')
    >>> df_5m = av.intraday('AAPL', interval='5min')
    >>> df_month = av.intraday_month('TSLA', month='2025-01')
    >>> universe = av.fetch_universe(['AAPL','MSFT','NVDA'], interval='1min')
    """

    def __init__(
        self,
        api_key:   Optional[str] = None,
        cache_dir: str | pathlib.Path = ".av_cache",
        rate_limit: float = 12.0,
        timeout:    int   = 30,
    ) -> None:
        self.api_key    = api_key or os.getenv("AV_API_KEY", AV_API_KEY)
        self.cache      = _FileCache(cache_dir)
        self.rate_limit = rate_limit
        self.timeout    = timeout
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intraday(
        self,
        symbol:     str,
        interval:   str = "1min",
        outputsize: str = "full",
        adjusted:   bool = True,
        extended:   bool = False,
        cache_ttl:  int  = 5,
    ) -> pd.DataFrame:
        """
        Fetch most-recent intraday OHLCV.

        Parameters
        ----------
        symbol     : ticker symbol (e.g. 'AAPL')
        interval   : '1min', '5min', '15min', '30min', '60min'
        outputsize : 'compact' (100 bars) | 'full' (up to 30 days)
        adjusted   : use split/dividend-adjusted prices
        extended   : include pre/post-market data
        cache_ttl  : cache lifetime in minutes (0 = no cache)

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume
                      DatetimeIndex (UTC), sorted ascending
        """
        params = {
            "function":      "TIME_SERIES_INTRADAY",
            "symbol":        symbol.upper(),
            "interval":      interval,
            "outputsize":    outputsize,
            "adjusted":      "true" if adjusted else "false",
            "extended_hours":"true" if extended else "false",
            "datatype":      "json",
            "apikey":        self.api_key,
        }
        data = self._get(params, cache_ttl)
        ts_key = _INTERVAL_KEY.get(interval, f"Time Series ({interval})")
        return self._parse_ts(data, ts_key)

    def intraday_month(
        self,
        symbol:    str,
        month:     str,          # format: 'YYYY-MM'
        interval:  str = "1min",
        adjusted:  bool = True,
        extended:  bool = False,
        cache_ttl: int  = 60 * 24,  # historical → cache 24h
    ) -> pd.DataFrame:
        """
        Fetch a specific calendar month of intraday data.
        Requires premium plan for months older than the latest 2.

        Parameters
        ----------
        month : 'YYYY-MM' (e.g. '2025-01')
        """
        params = {
            "function":       "TIME_SERIES_INTRADAY",
            "symbol":         symbol.upper(),
            "interval":       interval,
            "month":          month,
            "outputsize":     "full",
            "adjusted":       "true" if adjusted else "false",
            "extended_hours": "true" if extended else "false",
            "datatype":       "json",
            "apikey":         self.api_key,
        }
        data = self._get(params, cache_ttl)
        ts_key = _INTERVAL_KEY.get(interval, f"Time Series ({interval})")
        return self._parse_ts(data, ts_key)

    def fetch_universe(
        self,
        symbols:    List[str],
        interval:   str = "1min",
        outputsize: str = "full",
        adjusted:   bool = True,
        extended:   bool = False,
        cache_ttl:  int  = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch a universe of tickers, honouring rate limits between calls.

        Returns
        -------
        Dict[ticker, DataFrame] — tickers with errors are omitted.
        """
        universe: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.intraday(
                    sym, interval=interval, outputsize=outputsize,
                    adjusted=adjusted, extended=extended,
                    cache_ttl=cache_ttl,
                )
                if len(df) >= 1:
                    universe[sym] = df
                    logger.info("Fetched %s: %d bars", sym, len(df))
                else:
                    logger.warning("Empty data for %s — skipping", sym)
            except Exception as e:
                logger.error("Error fetching %s: %s", sym, e)
        return universe

    def latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the latest global quote for a symbol.

        Returns
        -------
        dict with keys: symbol, price, open, high, low, volume, change_pct
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol":   symbol.upper(),
            "apikey":   self.api_key,
        }
        data  = self._get(params, cache_ttl=1)
        quote = data.get("Global Quote", {})
        return {
            "symbol":     quote.get("01. symbol", symbol),
            "price":      float(quote.get("05. price",        0)),
            "open":       float(quote.get("02. open",         0)),
            "high":       float(quote.get("03. high",         0)),
            "low":        float(quote.get("04. low",          0)),
            "volume":     int(  quote.get("06. volume",       0)),
            "change_pct": float(quote.get("10. change percent", "0%").replace("%", "")),
        }

    def search(self, keywords: str) -> List[Dict[str, str]]:
        """Search for tickers matching keywords."""
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "apikey":   self.api_key,
        }
        data = self._get(params, cache_ttl=60)
        return data.get("bestMatches", [])

    def options_chain(
        self,
        symbol:    str,
        date:      Optional[str] = None,  # 'YYYY-MM-DD'; None = latest
        cache_ttl: int = 60,              # options data is slow-moving
    ) -> pd.DataFrame:
        """
        Fetch options chain from Alpha Vantage HISTORICAL_OPTIONS endpoint.

        Parameters
        ----------
        symbol    : equity ticker (e.g. 'AAPL')
        date      : 'YYYY-MM-DD' for historical chain; None = latest available
        cache_ttl : cache lifetime in minutes

        Returns
        -------
        pd.DataFrame with columns (all lowercased):
          strike, expiration, optiontype, openinterest, gamma,
          delta, impliedvolatility, volume, last
        Returns an empty DataFrame on any error.
        """
        params: Dict[str, Any] = {
            "function": "HISTORICAL_OPTIONS",
            "symbol":   symbol.upper(),
            "apikey":   self.api_key,
        }
        if date is not None:
            params["date"] = date

        try:
            data = self._get(params, cache_ttl=cache_ttl)
        except Exception as e:
            logger.error("options_chain(%s) fetch failed: %s", symbol, e)
            return pd.DataFrame()

        # AV returns data under key 'data' as a list of dicts
        records = data.get("data", [])
        if not records:
            # Try alternate key structure
            for key in data:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    records = data[key]
                    break

        if not records:
            logger.warning("options_chain(%s): empty response", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Normalize column names
        df.columns = [
            c.lower().replace(" ", "").replace("_", "") for c in df.columns
        ]

        # Standardize key column names expected by GammaSurface
        rename_map = {
            "contractid":        "contractid",
            "type":              "optiontype",
            "putcall":           "optiontype",
            "option_type":       "optiontype",
            "expire_date":       "expiration",
            "expiry":            "expiration",
            "expiration_date":   "expiration",
            "open_interest":     "openinterest",
            "impliedvolatility": "impliedvolatility",
            "iv":                "impliedvolatility",
        }
        df = df.rename(columns=rename_map)

        # Coerce numeric columns
        for col in ["strike", "gamma", "delta", "openinterest",
                    "impliedvolatility", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # MTF convenience helpers
    # ------------------------------------------------------------------

    def intraday_mtf(
        self,
        symbol: str,
        cache_ttl: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch 1m, 5m, 15m bars for multi-timeframe feature alignment.

        Returns
        -------
        {'1min': df, '5min': df, '15min': df}
        """
        result: Dict[str, pd.DataFrame] = {}
        for iv in ["1min", "5min", "15min"]:
            try:
                result[iv] = self.intraday(symbol, interval=iv, cache_ttl=cache_ttl)
            except Exception as e:
                logger.error("MTF fetch error %s %s: %s", symbol, iv, e)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, params: Dict, cache_ttl: int = 5) -> Dict:
        """Make a rate-limited, cached GET request."""
        if cache_ttl > 0:
            cached = self.cache.get(params, ttl_minutes=cache_ttl)
            if cached is not None:
                return cached

        # Rate limit
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            sleep_s = self.rate_limit - elapsed
            logger.debug("Rate limit: sleeping %.1fs", sleep_s)
            time.sleep(sleep_s)

        resp = requests.get(AV_BASE_URL, params=params, timeout=self.timeout)
        self._last_call = time.time()

        resp.raise_for_status()
        data = resp.json()

        # Check for AV error messages
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            logger.warning("Alpha Vantage API note: %s", data["Note"])
        if "Information" in data:
            raise RuntimeError(
                f"Alpha Vantage rate limit reached: {data['Information']}"
            )

        if cache_ttl > 0:
            self.cache.set(params, data)

        return data

    @staticmethod
    def _parse_ts(data: Dict, ts_key: str) -> pd.DataFrame:
        """Parse a time-series response dict into a clean OHLCV DataFrame."""
        ts = data.get(ts_key)
        if not ts:
            avail = [k for k in data if k != "Meta Data"]
            raise KeyError(
                f"Expected key '{ts_key}' not found in response. "
                f"Available: {avail}"
            )

        rows = []
        for dt_str, vals in ts.items():
            row = {"timestamp": pd.Timestamp(dt_str)}
            for av_col, our_col in _OHLCV_RENAME.items():
                row[our_col] = float(vals.get(av_col, 0))
            rows.append(row)

        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Ensure correct dtypes
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)

        return df


# ──────────────────────────────────────────────────────────────────────────────
# MCP helper (for Claude Code / MCP tool integration)
# ──────────────────────────────────────────────────────────────────────────────

class AlphaVantageMCP:
    """
    Thin wrapper for the Alpha Vantage MCP server at mcp.alphavantage.co.

    The MCP server exposes the same API surface as the REST API but via
    the Model Context Protocol, enabling tool use in Claude agents.

    Authentication: Bearer token in Authorization header.
    """

    BASE_URL = AV_MCP_URL
    TOKEN    = AV_MCP_TOKEN

    @classmethod
    def headers(cls) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {cls.TOKEN}",
            "Content-Type":  "application/json",
        }

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout:   int = 30,
    ) -> Any:
        """
        Call an MCP tool by name.

        Example tools (mirrors AV REST functions):
          get_time_series_intraday  — same as TIME_SERIES_INTRADAY
          get_global_quote          — same as GLOBAL_QUOTE
          search_symbols            — same as SYMBOL_SEARCH
        """
        payload = {
            "method":    "tools/call",
            "params": {
                "name":      tool_name,
                "arguments": arguments,
            },
        }
        resp = requests.post(
            cls.BASE_URL + "tools/call",
            json=payload,
            headers=cls.headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def list_tools(cls, timeout: int = 10) -> List[Dict]:
        """Return available MCP tools from the server."""
        resp = requests.get(
            cls.BASE_URL + "tools",
            headers=cls.headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("tools", [])
