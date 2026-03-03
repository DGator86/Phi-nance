"""
phinance.strategies — Indicator strategy library.

15 production-quality indicators drawn from:
  * Wilder (1978), Appel (1979), Bollinger (2002)
  * stockstats (jealous/stockstats)
  * arkochhar/Technical-Indicators
  * Stock.Indicators (.NET / DaveSkender)
  * Finance-Python (alpha-miner)

Public API
----------
    from phinance.strategies import INDICATOR_CATALOG, compute_indicator, list_indicators
    from phinance.strategies.base import BaseIndicator

Sub-modules
-----------
  base              — Abstract BaseIndicator (compute, compute_with_defaults, _normalize)
  indicator_catalog — Central registry: INDICATOR_CATALOG, compute_indicator, list_indicators
  rsi               — RSI (Wilder SMMA smoothing)
  macd              — MACD histogram (12/26/9 EMA, TradingView convention)
  bollinger         — Bollinger Bands mean-reversion
  dual_sma          — Dual SMA cross-over
  ema               — Dual EMA cross-over (faster than Dual SMA)
  mean_reversion    — Z-score mean-reversion
  breakout          — Donchian Channel breakout
  buy_hold          — Constant +0.5 benchmark
  vwap              — Rolling VWAP deviation
  atr               — Normalised ATR volatility-regime
  stochastic        — Stochastic %D oscillator
  williams_r        — Williams %R oscillator
  cci               — Commodity Channel Index
  obv               — On-Balance Volume momentum
  psar              — Parabolic SAR trend-following
  params            — Default parameter grids (daily + intraday)
"""

from phinance.strategies.indicator_catalog import (
    INDICATOR_CATALOG,
    compute_indicator,
    list_indicators,
)
from phinance.strategies.base import BaseIndicator

__all__ = [
    "INDICATOR_CATALOG",
    "compute_indicator",
    "list_indicators",
    "BaseIndicator",
]
