"""
phinance.data.vendors.unusual_whales
=====================================

Unusual Whales REST API client for options flow, dark pool, market
sentiment, options analytics, sector/ETF flow, and macroeconomic data.

Authentication
--------------
Set ``UNUSUAL_WHALES_API_KEY`` in your environment (or pass ``api_key``
directly).  The key is sent as a Bearer token::

    Authorization: Bearer <api_key>

Endpoint categories
-------------------
  Options Flow
    fetch_flow_alerts(symbol, limit)         — Whale / unusual options trades
    fetch_ticker_flow(symbol, limit)         — All flow for a single ticker
    fetch_options_chain(symbol)              — Full chain with Greeks (GEX-ready)

  Dark Pool
    fetch_dark_pool(symbol, limit)           — Recent off-exchange prints
    fetch_dark_pool_ticker(symbol, limit)    — Dark pool history for a ticker

  Market & Breadth
    fetch_market_tide(symbol)                — Net premium / put-call / GEX
    fetch_market_overview()                  — Advance/decline, breadth, VIX
    fetch_market_movers(direction, limit)    — Top gainers / losers

  Sector & ETF
    fetch_etf_flow(limit)                    — ETF net premium flow rankings
    fetch_etf_holdings(symbol)               — Constituent weights for an ETF

  Options Analytics
    fetch_iv_rank(symbol)                    — IV rank & IV percentile
    fetch_oi_change(symbol, limit)           — OI changes by strike/expiry
    fetch_options_volume(symbol, limit)      — Volume heatmap by strike/expiry
    fetch_pc_ratio(symbol)                   — Put/call volume & OI ratio

  Short Interest
    fetch_short_interest(symbol)             — Short float, utilisation, cost

  Congressional & Insider
    fetch_congress_trades(limit)             — Disclosed congressional trades
    fetch_insider_trades(symbol, limit)      — SEC Form 4 insider transactions

  GEX adapter
    options_chain_for_gex(symbol)           — Chain normalised for GammaSurface

Usage
-----
    from phinance.data.vendors.unusual_whales import UnusualWhalesClient

    client = UnusualWhalesClient(api_key="...")

    flow   = client.fetch_flow_alerts(symbol="SPY", limit=50)
    chain  = client.options_chain_for_gex("SPY")
    iv     = client.fetch_iv_rank("AAPL")
    tide   = client.fetch_market_tide("SPY")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.unusualwhales.com"


class UnusualWhalesClient:
    """Comprehensive client for the Unusual Whales REST API.

    Parameters
    ----------
    api_key : str | None
        Bearer token.  Falls back to the ``UNUSUAL_WHALES_API_KEY``
        environment variable if not provided.
    timeout : int
        HTTP request timeout in seconds (default 10).
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

    # ═══════════════════════════════════════════════════════════════════════
    # OPTIONS FLOW
    # ═══════════════════════════════════════════════════════════════════════

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
            Optional ticker filter (e.g. ``"SPY"``). ``None`` = market-wide.
        limit : int
            Maximum rows to return (max 200).

        Returns
        -------
        pd.DataFrame
            Columns: ticker, expiry, strike, option_type, premium, volume,
            open_interest, implied_volatility, sentiment.
            Index: time (UTC, DatetimeTZDtype).
        """
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        raw = self._get("/api/option-trades/flow-alerts", params=params)
        return self._to_df(raw, time_col="time", label="flow_alerts", symbol=symbol)

    def fetch_ticker_flow(
        self,
        symbol: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """All recent options trades for a single ticker.

        Returns the same schema as :meth:`fetch_flow_alerts` but restricted
        to *symbol* and including all trades (not just flagged alerts).

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"TSLA"``).
        limit : int
            Maximum rows to return.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get(f"/api/option-trades/ticker/{symbol.upper()}", params=params)
        return self._to_df(raw, time_col="time", label="ticker_flow", symbol=symbol)

    def fetch_options_chain(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Full options chain with Greeks for a ticker.

        Returns all strikes / expiries with live Greeks (delta, gamma, theta,
        vega, rho), open interest, volume, implied volatility, and bid/ask.

        Parameters
        ----------
        symbol : str
            Underlying ticker (e.g. ``"SPY"``).

        Returns
        -------
        pd.DataFrame
            Columns include: strike, expiration, option_type, open_interest,
            volume, implied_volatility, delta, gamma, theta, vega, bid, ask.
        """
        raw = self._get(f"/api/options/chain/{symbol.upper()}")
        df = self._raw_to_df(raw)
        logger.info("UnusualWhales options_chain: %d rows for %s", len(df), symbol)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # DARK POOL
    # ═══════════════════════════════════════════════════════════════════════

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
            dark_pool_position.
            Index: time (UTC).
        """
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        raw = self._get("/api/darkpool/recent", params=params)
        return self._to_df(raw, time_col="time", label="dark_pool", symbol=symbol)

    def fetch_dark_pool_ticker(
        self,
        symbol: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Dark pool print history for a specific ticker.

        Parameters
        ----------
        symbol : str
            Ticker to query.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Same schema as :meth:`fetch_dark_pool`.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get(f"/api/darkpool/{symbol.upper()}", params=params)
        return self._to_df(raw, time_col="time", label="dark_pool_ticker", symbol=symbol)

    # ═══════════════════════════════════════════════════════════════════════
    # MARKET & BREADTH
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_market_tide(
        self,
        symbol: str = "SPY",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Market Tide sentiment metrics for a ticker.

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

    def fetch_market_overview(self, **kwargs: Any) -> Dict[str, Any]:
        """Overall market breadth and sentiment snapshot.

        Returns advance/decline data, new highs/lows, sector performance,
        VIX level, and aggregate put/call ratio.

        Returns
        -------
        dict
            Keys: advancing, declining, unchanged, new_highs, new_lows,
            vix, put_call_ratio, sector_flow, timestamp.
        """
        raw = self._get("/api/market/overview")
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        logger.info("UnusualWhales market_overview fetched")
        return data if isinstance(data, dict) else {}

    def fetch_market_movers(
        self,
        direction: str = "gainers",
        limit: int = 20,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Top market movers by price change.

        Parameters
        ----------
        direction : str
            ``"gainers"`` or ``"losers"`` (default ``"gainers"``).
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, price, change_pct, volume, market_cap.
        """
        params: Dict[str, Any] = {"direction": direction, "limit": limit}
        raw = self._get("/api/market/movers", params=params)
        return self._raw_to_df(raw)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTOR & ETF FLOW
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_etf_flow(
        self,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """ETF options flow rankings by net premium.

        Useful for identifying institutional directional bets in sector ETFs
        (XLF, XLE, XLK, etc.) and broad-market ETFs (SPY, QQQ, IWM).

        Parameters
        ----------
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, net_call_premium, net_put_premium,
            call_volume, put_volume, put_call_ratio, total_premium.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get("/api/etf/flow", params=params)
        return self._raw_to_df(raw)

    def fetch_etf_holdings(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Constituent holdings and weights for an ETF.

        Parameters
        ----------
        symbol : str
            ETF ticker (e.g. ``"XLK"``).

        Returns
        -------
        pd.DataFrame
            Columns: ticker, name, weight, shares, market_value.
        """
        raw = self._get(f"/api/etf/{symbol.upper()}/holdings")
        df = self._raw_to_df(raw)
        logger.info("UnusualWhales etf_holdings: %d holdings for %s", len(df), symbol)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # OPTIONS ANALYTICS
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_iv_rank(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """IV Rank and IV Percentile for a ticker.

        IV Rank compares current IV to its 52-week range.
        IV Percentile shows the fraction of days IV was below its current level.

        Parameters
        ----------
        symbol : str
            Ticker to query.

        Returns
        -------
        dict
            Keys: iv_rank, iv_percentile, iv_current, iv_high_52w,
            iv_low_52w, ticker, timestamp.
        """
        raw = self._get(f"/api/stock/{symbol.upper()}/options/iv-rank")
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        logger.info("UnusualWhales iv_rank fetched for %s", symbol)
        return data if isinstance(data, dict) else {}

    def fetch_oi_change(
        self,
        symbol: str,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Open interest changes by strike and expiry.

        Shows which strikes gained or lost the most open interest since the
        prior close — a proxy for positioning changes.

        Parameters
        ----------
        symbol : str
            Ticker to query.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: strike, expiration, option_type, oi_current,
            oi_prev, oi_change, oi_change_pct.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get(f"/api/stock/{symbol.upper()}/options/oi-change", params=params)
        return self._raw_to_df(raw)

    def fetch_options_volume(
        self,
        symbol: str,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Options volume heatmap by strike and expiry.

        Parameters
        ----------
        symbol : str
            Ticker to query.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: strike, expiration, option_type, volume,
            open_interest, implied_volatility.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get(f"/api/stock/{symbol.upper()}/options/volume", params=params)
        return self._raw_to_df(raw)

    def fetch_pc_ratio(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Put/call volume and open-interest ratios for a ticker.

        Parameters
        ----------
        symbol : str
            Ticker to query.

        Returns
        -------
        dict
            Keys: call_volume, put_volume, pc_volume_ratio,
            call_oi, put_oi, pc_oi_ratio, ticker, timestamp.
        """
        raw = self._get(f"/api/stock/{symbol.upper()}/options/pc-ratio")
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        logger.info("UnusualWhales pc_ratio fetched for %s", symbol)
        return data if isinstance(data, dict) else {}

    # ═══════════════════════════════════════════════════════════════════════
    # SHORT INTEREST
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_short_interest(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Short interest data for a ticker.

        Parameters
        ----------
        symbol : str
            Ticker to query.

        Returns
        -------
        dict
            Keys: short_float, short_ratio (days-to-cover),
            short_shares, utilization, cost_to_borrow,
            ticker, as_of_date.
        """
        raw = self._get(f"/api/stock/{symbol.upper()}/short-interest")
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        logger.info("UnusualWhales short_interest fetched for %s", symbol)
        return data if isinstance(data, dict) else {}

    # ═══════════════════════════════════════════════════════════════════════
    # CONGRESSIONAL & INSIDER TRADES
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_congress_trades(
        self,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Recently disclosed congressional stock trades.

        Sourced from mandatory STOCK Act filings.  Useful as a proxy for
        informed political/regulatory positioning.

        Parameters
        ----------
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: politician, party, chamber, ticker, transaction_type,
            amount_range, filed_date, traded_date, issuer_name.
        """
        params: Dict[str, Any] = {"limit": limit}
        raw = self._get("/api/congress/trades", params=params)
        return self._to_df(raw, time_col="filed_date", label="congress_trades")

    def fetch_insider_trades(
        self,
        symbol: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """SEC Form 4 insider transaction filings.

        Parameters
        ----------
        symbol : str | None
            Optional ticker filter. ``None`` returns market-wide recent filings.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, insider_name, title, transaction_type, shares,
            price, value, filing_date, transaction_date.
        """
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["ticker"] = symbol.upper()

        raw = self._get("/api/insider/trades", params=params)
        return self._to_df(
            raw, time_col="filing_date", label="insider_trades", symbol=symbol
        )

    # ═══════════════════════════════════════════════════════════════════════
    # GEX ADAPTER — normalises UW chain for GammaSurface
    # ═══════════════════════════════════════════════════════════════════════

    def options_chain_for_gex(self, symbol: str) -> pd.DataFrame:
        """Fetch and normalise an options chain for use with GammaSurface.

        Calls :meth:`fetch_options_chain` and renames columns to the
        canonical set expected by ``GammaSurface._compute_gex_profile()``:

        =====================  =====================================
        GammaSurface column    Unusual Whales source column(s)
        =====================  =====================================
        strike                 strike
        expiration             expiration / expiry / expiration_date
        option_type            option_type / optiontype / type
        open_interest          open_interest / openinterest / oi
        gamma                  gamma
        =====================  =====================================

        Parameters
        ----------
        symbol : str
            Underlying ticker (e.g. ``"SPY"``).

        Returns
        -------
        pd.DataFrame
            Columns: strike (float), expiration (str/date), option_type
            (``"call"``/``"put"``), open_interest (int), gamma (float).
            Returns an empty DataFrame if the chain cannot be fetched.
        """
        try:
            raw_chain = self.fetch_options_chain(symbol)
        except Exception as exc:
            logger.warning(
                "options_chain_for_gex: fetch failed for %s — %s", symbol, exc
            )
            return pd.DataFrame()

        if raw_chain.empty:
            return raw_chain

        df = raw_chain.copy()
        df.columns = [str(c).lower().replace(" ", "_").strip() for c in df.columns]

        rename_map: Dict[str, str] = {}
        # expiration
        for src in ("expiry", "expiration_date"):
            if src in df.columns and "expiration" not in df.columns:
                rename_map[src] = "expiration"
        # option_type
        for src in ("optiontype", "type", "cp_flag", "call_put"):
            if src in df.columns and "option_type" not in df.columns:
                rename_map[src] = "option_type"
        # open_interest
        for src in ("openinterest", "oi", "open_int"):
            if src in df.columns and "open_interest" not in df.columns:
                rename_map[src] = "open_interest"

        if rename_map:
            df = df.rename(columns=rename_map)

        # Normalise option_type to lowercase 'call'/'put'
        if "option_type" in df.columns:
            df["option_type"] = (
                df["option_type"]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({"c": "call", "p": "put"})
            )

        # Keep only the columns GammaSurface needs, plus any extras
        gex_cols = ["strike", "expiration", "option_type", "open_interest", "gamma"]
        keep = [c for c in gex_cols if c in df.columns]
        extra = [c for c in df.columns if c not in gex_cols]
        df = df[keep + extra]

        # Coerce numeric types
        for col in ("strike", "open_interest", "gamma"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "options_chain_for_gex: %d rows ready for GammaSurface (%s)",
            len(df),
            symbol,
        )
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
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

    @staticmethod
    def _raw_to_df(raw: Any) -> pd.DataFrame:
        """Convert raw API response (dict with 'data' key or list) to DataFrame."""
        data: Any
        if isinstance(raw, dict):
            data = raw.get("data", raw)
        else:
            data = raw
        if not data:
            return pd.DataFrame()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame()

    def _to_df(
        self,
        raw: Any,
        time_col: str = "time",
        label: str = "",
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert raw API response to a time-indexed DataFrame."""
        df = self._raw_to_df(raw)
        if df.empty:
            return df

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.set_index(time_col).sort_index()

        if label:
            logger.info(
                "UnusualWhales %s: %d rows%s",
                label,
                len(df),
                f" for {symbol}" if symbol else "",
            )
        return df
