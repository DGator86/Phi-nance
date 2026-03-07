"""Options market data access stubs."""

from __future__ import annotations

from typing import Any, Dict


def fetch_options_market_data(symbol: str, as_of: str | None = None) -> Dict[str, Any]:
    """Return an empty payload placeholder for future options-chain integrations."""
    return {"symbol": symbol, "as_of": as_of, "contracts": []}
