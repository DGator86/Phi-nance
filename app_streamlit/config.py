"""UI configuration constants for the Streamlit live workbench."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Tuple

DEFAULT_SYMBOL = "SPY"
DEFAULT_TIMEFRAME = "1D"
DEFAULT_VENDOR = "alphavantage"
DEFAULT_TRADING_MODE = "equities"
DEFAULT_INITIAL_CAPITAL = 100_000.0

TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "1H", "1D"]
VENDOR_OPTIONS = ["alphavantage", "yfinance", "polygon"]
TRADING_MODE_OPTIONS = ["equities", "options"]
BLEND_METHOD_OPTIONS = ["weighted_sum", "majority_vote", "regime_weighted"]

DEFAULT_START_DATE = date.today() - timedelta(days=365)
DEFAULT_END_DATE = date.today()


@dataclass(frozen=True)
class IndicatorSpec:
    """Descriptor for rendering an indicator toggle and parameter controls."""

    description: str
    params: Dict[str, Tuple[float, float, float, float]]


INDICATOR_SPECS: Dict[str, IndicatorSpec] = {
    "RSI": IndicatorSpec(
        description="Relative Strength Index (momentum oscillator).",
        params={
            "rsi_period": (2, 50, 14, 1),
            "oversold": (10, 50, 30, 1),
            "overbought": (50, 95, 70, 1),
        },
    ),
    "MACD": IndicatorSpec(
        description="MACD crossover and histogram momentum.",
        params={
            "fast_period": (2, 50, 12, 1),
            "slow_period": (10, 100, 26, 1),
            "signal_period": (2, 30, 9, 1),
        },
    ),
    "Bollinger": IndicatorSpec(
        description="Bollinger band mean reversion.",
        params={"bb_period": (5, 100, 20, 1), "num_std": (1, 4, 2, 0.1)},
    ),
    "Dual SMA": IndicatorSpec(
        description="Fast/slow SMA crossover trend following.",
        params={"fast_period": (2, 100, 10, 1), "slow_period": (10, 300, 50, 1)},
    ),
    "Mean Reversion": IndicatorSpec(
        description="Distance from rolling SMA.",
        params={"sma_period": (5, 200, 20, 1)},
    ),
    "Breakout": IndicatorSpec(
        description="Donchian channel breakout signal.",
        params={"channel_period": (5, 100, 20, 1)},
    ),
    "Buy & Hold": IndicatorSpec(description="Baseline bullish exposure.", params={}),
    "VWAP": IndicatorSpec(
        description="VWAP deviation mean reversion (intraday).",
        params={"band_pct": (0.1, 3.0, 0.5, 0.1)},
    ),
}
