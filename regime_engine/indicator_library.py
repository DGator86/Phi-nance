"""
Indicator Library — Normalized downstream indicators with regime validity sets.

Each indicator:
  1. Computes a normalized signal from OHLCV only.
  2. Declares its indicator type (A/B/C/D).
  3. Declares a validity set V_j — list of taxonomy patterns that indicate
     when this indicator is meaningful.

Indicator types
---------------
  A — Bounded Oscillator  : output ∈ (-1, 1) via logit/sigmoid
  B — Unbounded Momentum  : output is a raw z-score / normalized value
  C — Discrete State      : output is flip probability ∈ [0, 1]
  D — Price Level         : output is normalized deviation from reference

Validity patterns
-----------------
  Patterns match on any of:
    - kingdom name  (e.g. 'NDR' → all NDR species)
    - 'kingdom.phylum.class_'  (e.g. 'DIR.*.TE')
    - specific species id (e.g. 'S08')

  Weights are resolved against species probabilities in the mixer.

Note: indicators here are DOWNSTREAM tools — they are NEVER used as inputs to
the regime inference taxonomy.  Regime must be inferred from base statistical
features only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseIndicator(ABC):
    """Abstract base for all indicators in the library."""

    #: Indicator type (A / B / C / D)
    indicator_type: str = "B"

    #: Validity patterns — list of strings (kingdom, KPC path, or species ID)
    validity_patterns: List[str] = []

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Compute normalized signal from OHLCV.

        Parameters
        ----------
        ohlcv : DataFrame with columns open/high/low/close/volume

        Returns
        -------
        pd.Series — same index as ohlcv, values normalized per type
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Simple rolling z-score."""
    mu  = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std().clip(lower=1e-10)
    return (series - mu) / std


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def _to_bounded(signal: pd.Series) -> pd.Series:
    """Map z-score to (-1,1) via tanh (Type A output)."""
    return np.tanh(signal / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Type A — Bounded Oscillators
# ──────────────────────────────────────────────────────────────────────────────

class RSI(BaseIndicator):
    """
    Relative Strength Index, mapped to (-1,1).
    Valid in: ranging regimes, trend-exhaustion species.
    """
    indicator_type = "A"
    validity_patterns = ["NDR", "DIR.*.TE", "TRN.*.FB"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close  = ohlcv["close"].astype(float)
        period = self.params.get("period", 14)
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss   = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
        rs     = gain / (loss + 1e-10)
        rsi    = 100.0 - 100.0 / (1.0 + rs)
        # Center on 50, map to (-1, 1)
        return _to_bounded((rsi - 50.0) / 25.0)


class Stochastic(BaseIndicator):
    """
    Stochastic %K oscillator, mapped to (-1,1).
    Valid in: ranging and exhaustion regimes.
    """
    indicator_type = "A"
    validity_patterns = ["NDR", "DIR.*.TE"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        high  = ohlcv["high"].astype(float)
        low   = ohlcv["low"].astype(float)
        close = ohlcv["close"].astype(float)
        k     = self.params.get("k_period", 14)
        d     = self.params.get("d_period", 3)

        roll_low  = low.rolling(k, min_periods=k // 2).min()
        roll_high = high.rolling(k, min_periods=k // 2).max()
        stoch_k   = (close - roll_low) / (roll_high - roll_low + 1e-10) * 100.0
        stoch_d   = stoch_k.rolling(d).mean()
        # Center on 50, map to (-1, 1)
        return _to_bounded((stoch_d - 50.0) / 25.0)


class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow — accumulation/distribution signal.
    Valid in: accumulation/distribution range species.
    """
    indicator_type = "A"
    validity_patterns = ["NDR.*.AR", "S13", "S17", "S21"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        high   = ohlcv["high"].astype(float)
        low    = ohlcv["low"].astype(float)
        close  = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float).clip(lower=1.0)
        period = self.params.get("period", 20)

        clv    = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv    = clv * volume
        cmf    = (
            mfv.rolling(period, min_periods=period // 2).sum()
            / volume.rolling(period, min_periods=period // 2).sum().clip(lower=1e-10)
        )
        return _to_bounded(cmf * 2.0)   # scale: CMF ∈ (-1,1) already


class RangePosition(BaseIndicator):
    """
    Price position within rolling range, centered at 0.
    Valid in: all ranging regimes.
    """
    indicator_type = "A"
    validity_patterns = ["NDR"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        high   = ohlcv["high"].astype(float)
        low    = ohlcv["low"].astype(float)
        close  = ohlcv["close"].astype(float)
        period = self.params.get("period", 20)

        r_low  = low.rolling(period, min_periods=period // 2).min()
        r_high = high.rolling(period, min_periods=period // 2).max()
        pos    = (close - r_low) / (r_high - r_low + 1e-10)  # ∈ [0,1]
        return _to_bounded((pos - 0.5) * 4.0)   # center + scale → (-1,1)


# ──────────────────────────────────────────────────────────────────────────────
# Type B — Unbounded Momentum
# ──────────────────────────────────────────────────────────────────────────────

class MACD(BaseIndicator):
    """
    MACD histogram — normalized momentum signal.
    Valid in: directional trend species.
    """
    indicator_type = "B"
    validity_patterns = ["DIR.*.PT", "DIR.*.PX"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close  = ohlcv["close"].astype(float)
        fast   = self.params.get("fast", 12)
        slow   = self.params.get("slow", 26)
        signal = self.params.get("signal", 9)

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd     = ema_fast - ema_slow
        sig_line = macd.ewm(span=signal, adjust=False).mean()
        hist     = macd - sig_line
        return _zscore(hist, window=60)


class RateOfChange(BaseIndicator):
    """
    Rate of Change (ROC) — normalized momentum.
    Valid in: directional trend and breakout species.
    """
    indicator_type = "B"
    validity_patterns = ["DIR", "TRN.*.SR", "TRN.*.RB"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close  = ohlcv["close"].astype(float)
        period = self.params.get("period", 10)
        roc    = (close / close.shift(period) - 1.0) * 100.0
        return _zscore(roc, window=60)


class Momentum(BaseIndicator):
    """
    Raw price momentum (close - close[n]) normalized.
    Valid in: persistent trend species.
    """
    indicator_type = "B"
    validity_patterns = ["DIR.*.PT", "DIR.*.PX"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close  = ohlcv["close"].astype(float)
        period = self.params.get("period", 10)
        mom    = close - close.shift(period)
        return _zscore(mom, window=60)


class ATRRatio(BaseIndicator):
    """
    ATR relative to its moving average — volatility expansion signal.
    Valid in: all regimes (universal volatility context).
    """
    indicator_type = "B"
    validity_patterns = ["DIR", "NDR", "TRN"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        high     = ohlcv["high"].astype(float)
        low      = ohlcv["low"].astype(float)
        close    = ohlcv["close"].astype(float)
        atr_p    = self.params.get("atr_period", 14)
        ma_p     = self.params.get("ma_period", 50)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr    = tr.ewm(span=atr_p, adjust=False).mean()
        atr_ma = atr.rolling(ma_p, min_periods=ma_p // 2).mean()
        ratio  = atr / (atr_ma + 1e-10)
        return _zscore(ratio, window=60)


# ──────────────────────────────────────────────────────────────────────────────
# Type C — Discrete State
# ──────────────────────────────────────────────────────────────────────────────

class TrendFlipProbability(BaseIndicator):
    """
    Probability that the next bar will flip sign (discrete state change).
    Based on rolling flip-rate momentum.
    Valid in: exhaustion and transition species.
    """
    indicator_type = "C"
    validity_patterns = ["DIR.*.TE", "TRN.*.FB", "S11", "S21", "S26"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close    = ohlcv["close"].astype(float)
        log_ret  = np.log(close / close.shift(1))
        sign_chg = (np.sign(log_ret) != np.sign(log_ret.shift(1))).astype(float)
        # Rolling flip rate is itself a probability ∈ [0,1]
        return sign_chg.rolling(20, min_periods=5).mean().fillna(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Type D — Price Level
# ──────────────────────────────────────────────────────────────────────────────

class VWAPDeviation(BaseIndicator):
    """
    Normalized deviation from session VWAP.
    Approximated as deviation from rolling VWAP over recent bars.
    Valid in: range and accumulation species.
    """
    indicator_type = "D"
    validity_patterns = ["NDR", "DIR.*.PT"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        high    = ohlcv["high"].astype(float)
        low     = ohlcv["low"].astype(float)
        close   = ohlcv["close"].astype(float)
        volume  = ohlcv["volume"].astype(float).clip(lower=1.0)
        session = self.params.get("session_minutes", 390)

        typical   = (high + low + close) / 3.0
        cum_tpv   = (typical * volume).rolling(session, min_periods=1).sum()
        cum_vol   = volume.rolling(session, min_periods=1).sum().clip(lower=1.0)
        vwap      = cum_tpv / cum_vol
        deviation = (close - vwap) / (vwap + 1e-10)
        return _zscore(deviation, window=60)


class VolumeProfileDeviation(BaseIndicator):
    """
    How far current price is from the rolling volume-weighted average level.
    Valid in: accumulation/distribution and range species.
    """
    indicator_type = "D"
    validity_patterns = ["NDR.*.AR", "NDR.*.BR"]

    def compute(self, ohlcv: pd.DataFrame) -> pd.Series:
        close   = ohlcv["close"].astype(float)
        volume  = ohlcv["volume"].astype(float).clip(lower=1.0)
        period  = self.params.get("period", 30)

        vwma    = (close * volume).rolling(period, min_periods=period // 2).sum() / \
                  volume.rolling(period, min_periods=period // 2).sum().clip(lower=1e-10)
        dev     = (close - vwma) / (vwma + 1e-10)
        return _zscore(dev, window=60)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

INDICATOR_CLASSES: Dict[str, type] = {
    "rsi":               RSI,
    "stochastic":        Stochastic,
    "cmf":               ChaikinMoneyFlow,
    "range_pos":         RangePosition,
    "macd":              MACD,
    "roc":               RateOfChange,
    "momentum":          Momentum,
    "atr_ratio":         ATRRatio,
    "trend_flip_prob":   TrendFlipProbability,
    "vwap_dev":          VWAPDeviation,
    "vol_profile_dev":   VolumeProfileDeviation,
}


def build_indicator(name: str, params: Dict[str, Any]) -> BaseIndicator:
    """Instantiate an indicator by name with given params dict."""
    cls = INDICATOR_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown indicator '{name}'. "
            f"Available: {list(INDICATOR_CLASSES)}"
        )
    return cls(params)
