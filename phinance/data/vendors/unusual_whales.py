"""
phinance.data.vendors.unusual_whales
=====================================

Unusual Whales REST API client for options flow, dark pool, and market
sentiment data.

Authentication
--------------
Set ``UNUSUAL_WHALES_API_KEY`` in your environment (or pass ``api_key``
directly).  The key is sent as a Bearer token::

    Authorization: Bearer <api_key>

Data types
----------
  flow_alerts   — Options flow / whale trades (``/api/option-trades/flow-alerts``)
  dark_pool     — Dark pool / off-exchange prints (``/api/darkpool/recent``)
  market_tide   — Market tide sentiment metrics (``/api/market/tide``)

Usage
-----
    from phinance.data.vendors.unusual_whales import UnusualWhalesClient

    client = UnusualWhalesClient(api_key="...")

    flow = client.fetch_flow_alerts(symbol="SPY", limit=50)
    dp   = client.fetch_dark_pool(symbol="SPY", limit=50)
    tide = client.fetch_market_tide(symbol="SPY")
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.unusualwhales.com"


class UnusualWhalesClient:
    """Lightweight client for the Unusual Whales REST API.

    Parameters
    ----------
    api_key : str | None
        Bearer token.  Falls back to the ``UNUSUAL_WHALES_API_KEY``
        environment variable if not provided.
    timeout : int
        HTTP request timeout in seconds.
    """

    name = "unusual_whales"

    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        self.api_key = api_key or os.environ.get("UNUSUAL_WHALES_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Unusual Whales API key is required. "
                "Pass api_key= or set UNUSUAL_WHALES_API_KEY."
            )
        self.timeout = timeout

    # ── Public methods ────────────────────────────────────────────────────────

    def fetch_flow_alerts(
        self,
        symbol: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch unusual options flow / whale trade alerts.

        Parameters
        ----------
        symbol : str | None
            Optional ticker to filter (e.g. ``"SPY"``). ``None`` returns
            market-wide flow.
        limit : int
            Maximum rows to return (default 100, max 200).

        Returns
        -------
        pd.DataFrame
            Columns: ticker, expiry, strike, option_type, premium,
            volume, open_interest, implied_volatility, sentiment,
            time (UTC).
        """
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        raw = self._get("/api/option-trades/flow-alerts", params=params)
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
        logger.info(
            "UnusualWhales flow_alerts: %d rows%s",
            len(df),
            f" for {symbol}" if symbol else "",
        )
        return df

    def fetch_dark_pool(
        self,
        symbol: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch recent dark pool / off-exchange block prints.

        Parameters
        ----------
        symbol : str | None
            Optional ticker filter.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, price, size, premium, exchange,
            dark_pool_position, time (UTC).
        """
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        raw = self._get("/api/darkpool/recent", params=params)
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
        logger.info(
            "UnusualWhales dark_pool: %d rows%s",
            len(df),
            f" for {symbol}" if symbol else "",
        )
        return df

    def fetch_market_tide(
        self,
        symbol: str = "SPY",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Market Tide sentiment metrics for a ticker.

        Includes net premium, put/call ratios, and gamma exposure.

        Parameters
        ----------
        symbol : str
            Ticker (default ``"SPY"``).

        Returns
        -------
        dict
            Keys: net_call_premium, net_put_premium, call_volume,
            put_volume, put_call_ratio, gamma_exposure, timestamp.
        """
        raw = self._get(f"/api/market/tide/{symbol.upper()}")
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        logger.info("UnusualWhales market_tide fetched for %s", symbol)
        return data if isinstance(data, dict) else {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute an authenticated GET request."""
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "requests is required for UnusualWhalesClient. "
                "Install it with: pip install requests"
            ) from exc

        url = f"{_BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        response = requests.get(
            url, headers=headers, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
