"""Base abstractions for order flow data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


ORDER_FLOW_COLUMNS = [
    "buy_volume",
    "sell_volume",
    "cumulative_delta",
    "tick_count",
    "bid",
    "ask",
    "spread",
    "depth",
]


class OrderFlowProvider(ABC):
    """Abstract base class for per-bar order flow data providers."""

    @abstractmethod
    def get_order_flow(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return order flow data aligned to ``ohlcv.index``."""


def ensure_order_flow_schema(flow: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Ensure an order flow DataFrame contains all standard columns."""
    out = flow.reindex(index=index).copy()
    for col in ORDER_FLOW_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out[ORDER_FLOW_COLUMNS]
