"""
phinance.options.market_data
=============================

MarketDataApp connector for live options chain snapshots.

Provides ``get_marketdataapp_snapshot()`` — a thin convenience wrapper
that returns an ``OptionsSnapshot`` for the near-ATM option of a given
symbol, or ``None`` when the token is absent / API unreachable.

Usage
-----
    from phinance.options.market_data import get_marketdataapp_snapshot

    snap = get_marketdataapp_snapshot("SPY", spot_price=450.0)
    if snap:
        print(f"ATM delta: {snap.delta}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests


@dataclass
class OptionsSnapshot:
    """A single near-ATM options chain snapshot.

    Attributes
    ----------
    symbol            : Underlying ticker
    as_of             : Timestamp string of the snapshot
    strike            : Strike price of the near-ATM contract
    expiry            : Expiry date string
    option_type       : ``"call"`` or ``"put"``
    bid, ask, mid     : Quote prices
    delta             : Option delta (None if unavailable)
    implied_volatility: IV (None if unavailable)
    source            : Data provider name
    """

    symbol:             str
    as_of:              str
    strike:             float
    expiry:             str
    option_type:        str
    bid:                float
    ask:                float
    mid:                float
    delta:              Optional[float]
    implied_volatility: Optional[float]
    source:             str


class MarketDataAppClient:
    """Thin REST client for the MarketDataApp v1 options endpoint.

    Handles both columnar-array and list response formats.
    Falls back gracefully on 401/403/404 or network errors.
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = "https://api.marketdata.app/v1",
    ) -> None:
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}"}

    def get_near_atm_snapshot(
        self,
        symbol: str,
        spot_price: float,
        option_type: str = "call",
    ) -> Optional[OptionsSnapshot]:
        """Fetch the nearest-to-ATM option snapshot.

        Parameters
        ----------
        symbol      : str — underlying ticker (e.g. ``"SPY"``)
        spot_price  : float — current underlying price for ATM selection
        option_type : ``"call"`` | ``"put"``

        Returns
        -------
        OptionsSnapshot or None
        """
        expiry = (date.today() + timedelta(days=30)).isoformat()
        url = f"{self.base_url}/options/chain/{symbol.upper()}/"
        params = {
            "expiration": expiry,
            "side":       "call" if option_type == "call" else "put",
            "range":      "otm",
            "dateformat": "timestamp",
        }
        try:
            r = requests.get(
                url, headers=self._headers(), params=params, timeout=20
            )
        except Exception:
            return None

        if r.status_code in (401, 403, 404):
            return None
        try:
            r.raise_for_status()
        except Exception:
            return None

        payload = r.json()
        if isinstance(payload, dict) and isinstance(payload.get("strike"), list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            return None

        if df.empty or "strike" not in df.columns:
            return None

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"])
        if df.empty:
            return None

        idx = (df["strike"] - float(spot_price)).abs().idxmin()
        row = df.loc[idx]

        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("last", 0.0) or 0.0)

        return OptionsSnapshot(
            symbol            = symbol.upper(),
            as_of             = str(pd.Timestamp.utcnow()),
            strike            = float(row["strike"]),
            expiry            = str(row.get("expiration", expiry)),
            option_type       = option_type,
            bid               = bid,
            ask               = ask,
            mid               = float(mid),
            delta             = float(row["delta"]) if pd.notna(row.get("delta")) else None,
            implied_volatility= float(row["iv"]) if pd.notna(row.get("iv")) else None,
            source            = "marketdataapp",
        )


def get_marketdataapp_snapshot(
    symbol: str,
    spot_price: float,
    option_type: str = "call",
) -> Optional[OptionsSnapshot]:
    """Convenience wrapper: fetch snapshot using ``MARKETDATAAPP_API_TOKEN`` env var.

    Returns ``None`` when the token is absent or the API is unreachable.
    """
    token = os.environ.get("MARKETDATAAPP_API_TOKEN", "").strip()
    if not token:
        return None
    try:
        return MarketDataAppClient(token).get_near_atm_snapshot(
            symbol=symbol,
            spot_price=spot_price,
            option_type=option_type,
        )
    except Exception:
        return None
