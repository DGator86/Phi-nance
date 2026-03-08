"""Heuristic order flow provider derived from OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.indicators.orderflow.base import OrderFlowProvider, ensure_order_flow_schema
from phi.logging import get_logger

logger = get_logger(__name__)


class OHLCVOrderFlowProvider(OrderFlowProvider):
    """Estimate order flow fields from OHLCV data using bar-direction heuristics."""

    def __init__(self, estimation_method: str = "candle_direction") -> None:
        self.estimation_method = estimation_method

    def get_order_flow(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute heuristic buy/sell volume and liquidity proxy columns."""
        required = {"open", "high", "low", "close", "volume"}
        missing = sorted(required - set(ohlcv.columns))
        if missing:
            raise ValueError(f"OHLCV data missing required columns: {missing}")

        if self.estimation_method != "candle_direction":
            logger.warning("Unknown order flow estimation method '%s'; falling back to candle_direction.", self.estimation_method)

        open_ = ohlcv["open"].astype(float)
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float).clip(lower=0.0)

        bullish = close > open_
        bearish = close < open_
        neutral = ~(bullish | bearish)

        buy_volume = np.where(bullish, volume, np.where(neutral, volume * 0.5, 0.0))
        sell_volume = np.where(bearish, volume, np.where(neutral, volume * 0.5, 0.0))

        mid = ((high + low) / 2.0).replace(0.0, np.nan)
        spread_proxy = ((high - low) / mid).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        flow = pd.DataFrame(
            {
                "buy_volume": pd.Series(buy_volume, index=ohlcv.index),
                "sell_volume": pd.Series(sell_volume, index=ohlcv.index),
                "tick_count": 1.0,
                "bid": np.nan,
                "ask": np.nan,
                "spread": spread_proxy,
                "depth": volume,
            },
            index=ohlcv.index,
        )
        flow["cumulative_delta"] = (flow["buy_volume"] - flow["sell_volume"]).cumsum()
        return ensure_order_flow_schema(flow, ohlcv.index)
