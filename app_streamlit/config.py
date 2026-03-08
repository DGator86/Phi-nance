"""UI configuration constants for the Streamlit live workbench."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TypeAlias, TypedDict

from phi.logging import get_logger

logger = get_logger(__name__)


class SelectParamSpec(TypedDict):
    """Selectbox parameter metadata used by Streamlit controls."""

    type: str
    options: list[str]
    default: str


ParamRange: TypeAlias = tuple[float, float, float, float]
IndicatorParamSpec: TypeAlias = ParamRange | SelectParamSpec

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
    params: dict[str, IndicatorParamSpec]
    category: str = "Core"




def _select_param(options: list[str], default: str) -> SelectParamSpec:
    """Convenience helper for selectbox parameter specs."""
    return {"type": "select", "options": options, "default": default}


INDICATOR_SPECS: dict[str, IndicatorSpec] = {
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

    "Orderflow VWAP": IndicatorSpec(
        description="Order-flow VWAP deviation normalized by ATR.",
        params={"atr_period": (5, 50, 14, 1), "clip_value": (0.5, 5.0, 2.0, 0.5)},
        category="Order Flow & Liquidity",
    ),
    "Volume Profile": IndicatorSpec(
        description="Rolling point-of-control proximity from volume profile.",
        params={"window": (5, 120, 20, 1), "bins": (4, 40, 16, 1), "near_poc_threshold": (0.0005, 0.02, 0.002, 0.0005)},
        category="Order Flow & Liquidity",
    ),
    "Cumulative Delta": IndicatorSpec(
        description="Estimated buy-sell pressure from candle-direction volume split.",
        params={"window": (5, 100, 20, 1), "clip_value": (0.2, 3.0, 1.0, 0.1)},
        category="Order Flow & Liquidity",
    ),
    "Liquidity Metrics": IndicatorSpec(
        description="Spread proxy and Amihud illiquidity based signal.",
        params={"window": (5, 100, 20, 1), "amihud_scale": (1000, 10000000, 1000000, 1000)},
        category="Order Flow & Liquidity",
    ),


    "Rolling Entropy": IndicatorSpec(
        description="Shannon entropy over rolling return distributions.",
        params={"window": (5, 120, 20, 1), "bins": (2, 40, 10, 1)},
        category="Information Theory",
    ),
    "Mutual Information": IndicatorSpec(
        description="Dependency between returns and lagged returns.",
        params={"window": (10, 150, 30, 1), "bins": (2, 30, 8, 1), "lag": (1, 10, 1, 1)},
        category="Information Theory",
    ),
    "Fisher Information": IndicatorSpec(
        description="Standardized return-slope information proxy.",
        params={"window": (5, 120, 20, 1)},
        category="Information Theory",
    ),
    "KL Divergence": IndicatorSpec(
        description="Divergence between adjacent rolling return distributions.",
        params={"window": (10, 150, 30, 1), "bins": (2, 40, 10, 1)},
        category="Information Theory",
    ),

    "MFT Signal": IndicatorSpec(
        description="Simplified Market Field Theory gradient-direction signal.",
        params={
            "kernel": _select_param(["gaussian", "exp", "linear"], "gaussian"),
            "sigma": (1, 50, 10, 1),
            "threshold": (0.0, 5.0, 0.0, 0.05),
            "smooth_window": (1, 50, 1, 1),
        },
        category="Market Field Theory",
    ),
    "MFT Energy": IndicatorSpec(
        description="Relative field-energy signal (low activity bullish, high activity defensive).",
        params={
            "kernel": _select_param(["gaussian", "exp", "linear"], "gaussian"),
            "sigma": (1, 50, 10, 1),
            "energy_window": (3, 120, 20, 1),
        },
        category="Market Field Theory",
    ),
}
