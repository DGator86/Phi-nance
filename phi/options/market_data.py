"""Options market-data connectors used by options backtest mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import os
import pandas as pd
import requests


@dataclass
class OptionsSnapshot:
    symbol: str
    as_of: str
    strike: float
    expiry: str
    option_type: str
    bid: float
    ask: float
    mid: float
    delta: Optional[float]
    implied_volatility: Optional[float]
    source: str


class MarketDataAppClient:
    """Thin client for MarketDataApp options endpoint.

    API details can differ across tiers/endpoints; this client intentionally
    implements a resilient parser and falls back gracefully when unavailable.
    """

    def __init__(self, api_token: str, base_url: str = "https://api.marketdata.app/v1") -> None:
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}"}

    def get_near_atm_snapshot(self, symbol: str, spot_price: float, option_type: str = "call") -> Optional[OptionsSnapshot]:
        expiry = (date.today() + timedelta(days=30)).isoformat()
        url = f"{self.base_url}/options/chain/{symbol.upper()}/"
        params = {
            "expiration": expiry,
            "side": "call" if option_type == "call" else "put",
            "range": "otm",
            "dateformat": "timestamp",
        }
        r = requests.get(url, headers=self._headers(), params=params, timeout=20)
        if r.status_code in (401, 403, 404):
            return None
        r.raise_for_status()
        payload = r.json()

        # Endpoint returns columnar arrays in many versions.
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
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get("last", 0.0) or 0.0)

        return OptionsSnapshot(
            symbol=symbol.upper(),
            as_of=str(pd.Timestamp.utcnow()),
            strike=float(row["strike"]),
            expiry=str(row.get("expiration", expiry)),
            option_type="call" if option_type == "call" else "put",
            bid=bid,
            ask=ask,
            mid=float(mid),
            delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
            implied_volatility=float(row["iv"]) if pd.notna(row.get("iv")) else None,
            source="marketdataapp",
        )


def get_marketdataapp_snapshot(symbol: str, spot_price: float, option_type: str = "call") -> Optional[OptionsSnapshot]:
    token = os.environ.get("MARKETDATAAPP_API_TOKEN", "").strip()
    if not token:
        return None
    try:
        return MarketDataAppClient(token).get_near_atm_snapshot(symbol=symbol, spot_price=spot_price, option_type=option_type)
    except Exception:
        return None
