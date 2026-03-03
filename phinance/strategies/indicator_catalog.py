"""
phinance.strategies.indicator_catalog
======================================

Central registry of all available indicator strategies.

INDICATOR_CATALOG maps display names → BaseIndicator instances.
All indicator modules are imported here so new indicators only require:

  1. Create ``phinance/strategies/myindicator.py``
  2. Subclass ``BaseIndicator``
  3. Add one entry here

Public API
----------
  INDICATOR_CATALOG    — dict: {name: BaseIndicator instance}
  compute_indicator(name, df, params) → pd.Series
  list_indicators()    — list of indicator names

Registered indicators
---------------------
  RSI           — Wilder RSI mean-reversion
  MACD          — MACD histogram momentum
  Bollinger     — Bollinger Bands mean-reversion
  Dual SMA      — Dual SMA cross-over trend
  EMA Cross     — Dual EMA cross-over trend (faster)
  Mean Reversion— Z-score mean-reversion
  Breakout      — Donchian Channel breakout
  Buy & Hold    — Constant +0.5 benchmark
  VWAP          — Volume-Weighted Average Price deviation
  ATR           — Normalised ATR volatility-regime
  Stochastic    — Stochastic %D mean-reversion
  Williams %R   — Williams %R mean-reversion
  CCI           — Commodity Channel Index
  OBV           — On-Balance Volume momentum
  PSAR          — Parabolic SAR trend-following
  Aroon         — Aroon Oscillator trend strength
  Ulcer Index   — Downside-risk / drawdown severity
  KST           — Know Sure Thing multi-period momentum
  TRIX          — Triple Smoothed EMA momentum oscillator
  Mass Index    — High-low range expansion reversal indicator
  DEMA          — Double Exponential Moving Average trend
  TEMA          — Triple Exponential Moving Average trend
  KAMA          — Kaufman Adaptive Moving Average trend
  ZLEMA         — Zero Lag EMA trend
  HMA           — Hull Moving Average trend
  VWMA          — Volume Weighted Moving Average trend
  Ichimoku      — Ichimoku Kinko Hyo trend/momentum
  Donchian      — Donchian Channel position
  Keltner       — Keltner Channel breakout/volatility
  Elder Ray     — Elder Ray bull/bear power
  DPO           — Detrended Price Oscillator cycle
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.strategies.rsi           import RSIIndicator
from phinance.strategies.macd          import MACDIndicator
from phinance.strategies.bollinger     import BollingerIndicator
from phinance.strategies.dual_sma      import DualSMAIndicator
from phinance.strategies.ema           import EMACrossIndicator
from phinance.strategies.mean_reversion import MeanReversionIndicator
from phinance.strategies.breakout      import BreakoutIndicator
from phinance.strategies.buy_hold      import BuyHoldIndicator
from phinance.strategies.vwap          import VWAPIndicator
from phinance.strategies.atr           import ATRIndicator
from phinance.strategies.stochastic    import StochasticIndicator
from phinance.strategies.williams_r    import WilliamsRIndicator
from phinance.strategies.cci           import CCIIndicator
from phinance.strategies.obv           import OBVIndicator
from phinance.strategies.psar          import PSARIndicator
from phinance.strategies.aroon         import AroonIndicator
from phinance.strategies.ulcer_index   import UlcerIndexIndicator
from phinance.strategies.kst           import KSTIndicator
from phinance.strategies.trix          import TRIXIndicator
from phinance.strategies.mass_index    import MassIndexIndicator
from phinance.strategies.dema          import DEMAIndicator
from phinance.strategies.tema          import TEMAIndicator
from phinance.strategies.kama          import KAMAIndicator
from phinance.strategies.zlema         import ZLEMAIndicator
from phinance.strategies.hma           import HMAIndicator
from phinance.strategies.vwma          import VWMAIndicator
from phinance.strategies.ichimoku      import IchimokuIndicator
from phinance.strategies.donchian      import DonchianIndicator
from phinance.strategies.keltner       import KeltnerIndicator
from phinance.strategies.elder_ray     import ElderRayIndicator
from phinance.strategies.dpo           import DPOIndicator

# ── Registry ─────────────────────────────────────────────────────────────────

INDICATOR_CATALOG: Dict[str, Any] = {
    "RSI":            RSIIndicator(),
    "MACD":           MACDIndicator(),
    "Bollinger":      BollingerIndicator(),
    "Dual SMA":       DualSMAIndicator(),
    "EMA Cross":      EMACrossIndicator(),
    "Mean Reversion": MeanReversionIndicator(),
    "Breakout":       BreakoutIndicator(),
    "Buy & Hold":     BuyHoldIndicator(),
    "VWAP":           VWAPIndicator(),
    "ATR":            ATRIndicator(),
    "Stochastic":     StochasticIndicator(),
    "Williams %R":    WilliamsRIndicator(),
    "CCI":            CCIIndicator(),
    "OBV":            OBVIndicator(),
    "PSAR":           PSARIndicator(),
    "Aroon":          AroonIndicator(),
    "Ulcer Index":    UlcerIndexIndicator(),
    "KST":            KSTIndicator(),
    "TRIX":           TRIXIndicator(),
    "Mass Index":     MassIndexIndicator(),
    "DEMA":           DEMAIndicator(),
    "TEMA":           TEMAIndicator(),
    "KAMA":           KAMAIndicator(),
    "ZLEMA":          ZLEMAIndicator(),
    "HMA":            HMAIndicator(),
    "VWMA":           VWMAIndicator(),
    "Ichimoku":       IchimokuIndicator(),
    "Donchian":       DonchianIndicator(),
    "Keltner":        KeltnerIndicator(),
    "Elder Ray":      ElderRayIndicator(),
    "DPO":            DPOIndicator(),
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def list_indicators() -> List[str]:
    """Return all registered indicator names."""
    return list(INDICATOR_CATALOG.keys())


def compute_indicator(
    name: str,
    df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """Compute a named indicator's signal from OHLCV data.

    Parameters
    ----------
    name : str
        Indicator name (must be a key in ``INDICATOR_CATALOG``).
    df : pd.DataFrame
        OHLCV DataFrame with columns ``[open, high, low, close, volume]``
        and a DatetimeIndex.
    params : dict, optional
        Parameter overrides merged with the indicator's ``default_params``.

    Returns
    -------
    pd.Series — signal in [−1, 1]

    Raises
    ------
    phinance.exceptions.UnknownIndicatorError
        When *name* is not in the catalog.
    phinance.exceptions.IndicatorComputationError
        When the indicator's ``compute()`` raises.
    """
    from phinance.exceptions import UnknownIndicatorError

    indicator = INDICATOR_CATALOG.get(name)
    if indicator is None:
        raise UnknownIndicatorError(
            f"Unknown indicator: '{name}'. "
            f"Available: {list(INDICATOR_CATALOG.keys())}"
        )
    return indicator.compute_with_defaults(df, params or {})
