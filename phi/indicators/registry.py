"""
phi.indicators.registry — Indicator Registry
=============================================
Each indicator:
  - Has a unique name (key)
  - Has a display_name and description
  - Has a params dict: {param_key: {label, default, min, max, step, type}}
  - Has a compute(ohlcv, params) -> pd.Series method
  - Produces a normalized signal in [-1, +1] where:
      +1 = strong bullish
      -1 = strong bearish
       0 = neutral
"""

from __future__ import annotations

from phi.logging import get_logger

logger = get_logger(__name__)

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from phi.indicators.orderflow import (
    compute_cumulative_delta_signal,
    compute_liquidity_signal,
    compute_volume_profile_signal,
    compute_vwap_signal,
    get_order_flow_provider,
)

from phi.mft.signals import mft_energy_signal, mft_signal


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 3).mean()
    std = series.rolling(window, min_periods=window // 3).std().clip(lower=1e-10)
    return (series - mu) / std


def _to_signal(series: pd.Series) -> pd.Series:
    """Map any float series to [-1, +1] via tanh."""
    return np.tanh(series / 2.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = ohlcv["high"].astype(float)
    low   = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Compute functions: (ohlcv, params) → pd.Series in [-1, +1]
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    period = int(params.get("period", 14))
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
    rs     = gain / (loss + 1e-10)
    rsi    = 100.0 - 100.0 / (1.0 + rs)
    # RSI < 30 → bullish signal (+1), RSI > 70 → bearish (-1)
    signal = -((rsi - 50.0) / 25.0)
    return _to_signal(signal)


def _compute_macd(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    fast   = int(params.get("fast", 12))
    slow   = int(params.get("slow", 26))
    sig_p  = int(params.get("signal", 9))
    macd   = _ema(close, fast) - _ema(close, slow)
    sig    = _ema(macd, sig_p)
    hist   = macd - sig
    return _to_signal(_safe_zscore(hist, 60))


def _compute_bollinger(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    period = int(params.get("period", 20))
    nstd   = float(params.get("num_std", 2.0))
    ma     = close.rolling(period, min_periods=period // 2).mean()
    std    = close.rolling(period, min_periods=period // 2).std().clip(lower=1e-10)
    upper  = ma + nstd * std
    lower  = ma - nstd * std
    # Position within band: below lower = +1 (buy), above upper = -1 (sell)
    pos    = (close - lower) / (upper - lower + 1e-10)   # 0=at lower, 1=at upper
    signal = -(pos - 0.5) * 4.0                           # center and scale
    return _to_signal(signal)


def _compute_stochastic(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    high  = ohlcv["high"].astype(float)
    low   = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    k_p   = int(params.get("k_period", 14))
    d_p   = int(params.get("d_period", 3))
    r_low  = low.rolling(k_p, min_periods=k_p // 2).min()
    r_high = high.rolling(k_p, min_periods=k_p // 2).max()
    stoch  = (close - r_low) / (r_high - r_low + 1e-10) * 100.0
    stoch_d = stoch.rolling(d_p).mean()
    # Oversold < 20 → +1, Overbought > 80 → -1
    signal  = -((stoch_d - 50.0) / 25.0)
    return _to_signal(signal)


def _compute_dual_sma(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    fast   = int(params.get("fast", 10))
    slow   = int(params.get("slow", 50))
    sma_f  = close.rolling(fast, min_periods=fast // 2).mean()
    sma_s  = close.rolling(slow, min_periods=slow // 2).mean()
    diff   = (sma_f - sma_s) / (sma_s + 1e-10)
    return _to_signal(_safe_zscore(diff, 60))


def _compute_ema_crossover(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close = ohlcv["close"].astype(float)
    fast  = int(params.get("fast", 9))
    slow  = int(params.get("slow", 21))
    diff  = _ema(close, fast) - _ema(close, slow)
    norm  = diff / (close + 1e-10) * 100
    return _to_signal(_safe_zscore(norm, 60))


def _compute_momentum(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    period = int(params.get("period", 10))
    mom    = close.diff(period)
    return _to_signal(_safe_zscore(mom, 60))


def _compute_roc(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    close  = ohlcv["close"].astype(float)
    period = int(params.get("period", 10))
    roc    = (close / close.shift(period) - 1.0) * 100.0
    return _to_signal(_safe_zscore(roc, 60))


def _compute_atr_ratio(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """ATR ratio vs its MA — high = expanding volatility (ambiguous direction, use for sizing)."""
    atr_p = int(params.get("atr_period", 14))
    ma_p  = int(params.get("ma_period", 50))
    atr   = _atr(ohlcv, atr_p)
    ma    = atr.rolling(ma_p, min_periods=ma_p // 2).mean()
    ratio = (atr / (ma + 1e-10)) - 1.0
    # Expanding vol = 0 bias (neutral), contracting = small bullish bias
    return _to_signal(-ratio)


def _compute_vwap_dev(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    high   = ohlcv["high"].astype(float)
    low    = ohlcv["low"].astype(float)
    close  = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=1.0)
    period = int(params.get("period", 20))
    typical   = (high + low + close) / 3.0
    cum_tpv   = (typical * volume).rolling(period, min_periods=1).sum()
    cum_vol   = volume.rolling(period, min_periods=1).sum().clip(lower=1.0)
    vwap      = cum_tpv / cum_vol
    deviation = (close - vwap) / (vwap + 1e-10)
    # Below VWAP → buy bias, above → sell
    return _to_signal(-_safe_zscore(deviation, 60))


def _compute_cmf(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    high   = ohlcv["high"].astype(float)
    low    = ohlcv["low"].astype(float)
    close  = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=1.0)
    period = int(params.get("period", 20))
    clv    = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv    = clv * volume
    cmf    = (mfv.rolling(period, min_periods=period // 2).sum() /
              volume.rolling(period, min_periods=period // 2).sum().clip(lower=1e-10))
    return _to_signal(cmf * 2.0)


def _compute_adx(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """ADX-based: uses DI+/DI- for direction, ADX for strength."""
    high  = ohlcv["high"].astype(float)
    low   = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    period = int(params.get("period", 14))

    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    dm_plus  = np.where(up > down, up, 0.0)
    dm_minus = np.where(down > up, down, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr14  = pd.Series(tr).ewm(span=period, adjust=False).mean()
    di_plus  = pd.Series(dm_plus).ewm(span=period, adjust=False).mean() / (atr14 + 1e-10) * 100
    di_minus = pd.Series(dm_minus).ewm(span=period, adjust=False).mean() / (atr14 + 1e-10) * 100

    dx = ((di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)) * 100
    adx = dx.ewm(span=period, adjust=False).mean()

    direction = np.sign(di_plus.values - di_minus.values)
    strength  = (adx / 50.0).clip(0, 2)
    signal    = pd.Series(direction * strength, index=ohlcv.index)
    return _to_signal(signal)


def _compute_mft_signal(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """Simplified MFT directional signal based on potential gradient."""
    close = ohlcv["close"].astype(float)
    kernel = str(params.get("kernel", "gaussian"))
    sigma = float(params.get("sigma", 10.0))
    threshold = float(params.get("threshold", 0.0))
    smooth_window = int(params.get("smooth_window", 1))
    return mft_signal(close=close, kernel=kernel, sigma=sigma, threshold=threshold, smooth_window=smooth_window)


def _compute_mft_energy(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """MFT energy-derived signal from relative field activity."""
    close = ohlcv["close"].astype(float)
    kernel = str(params.get("kernel", "gaussian"))
    sigma = float(params.get("sigma", 10.0))
    energy_window = int(params.get("energy_window", 20))
    return mft_energy_signal(close=close, kernel=kernel, sigma=sigma, energy_window=energy_window)


def _compute_wyckoff(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """Simplified Wyckoff: accumulation/distribution proxy."""
    close  = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=1.0)
    lookback = int(params.get("lookback", 30))

    # Volume-weighted price momentum
    vwm   = (close * volume).rolling(lookback, min_periods=lookback // 2).sum()
    vol_s = volume.rolling(lookback, min_periods=lookback // 2).sum().clip(lower=1e-10)
    vwma  = vwm / vol_s
    dev   = (close - vwma) / (vwma + 1e-10)

    # Divergence: price making highs on declining volume = distribution
    price_change = close.diff(lookback)
    vol_change   = volume.diff(lookback)
    divergence   = price_change * np.sign(-vol_change)

    signal = _safe_zscore(dev, 60) + _safe_zscore(divergence, 60) * 0.3
    return _to_signal(signal)


def _compute_range_pos(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    """Price position within rolling range → mean-reversion signal."""
    high   = ohlcv["high"].astype(float)
    low    = ohlcv["low"].astype(float)
    close  = ohlcv["close"].astype(float)
    period = int(params.get("period", 20))
    r_low  = low.rolling(period, min_periods=period // 2).min()
    r_high = high.rolling(period, min_periods=period // 2).max()
    pos    = (close - r_low) / (r_high - r_low + 1e-10)
    signal = -(pos - 0.5) * 4.0
    return _to_signal(signal)



def _compute_orderflow_vwap(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    atr_period = int(params.get("atr_period", 14))
    clip_value = float(params.get("clip_value", 2.0))
    return compute_vwap_signal(ohlcv, atr_period=atr_period, clip_value=clip_value)


def _compute_orderflow_volume_profile(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    window = int(params.get("window", 20))
    bins = int(params.get("bins", 16))
    near_poc_threshold = float(params.get("near_poc_threshold", 0.002))
    _poc, signal = compute_volume_profile_signal(ohlcv, window=window, bins=bins, near_poc_threshold=near_poc_threshold)
    return signal


def _compute_orderflow_cumulative_delta(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    provider = get_order_flow_provider()
    flow = provider.get_order_flow(ohlcv)
    window = int(params.get("window", 20))
    clip_value = float(params.get("clip_value", 1.0))
    return compute_cumulative_delta_signal(flow, ohlcv["volume"], window=window, clip_value=clip_value)


def _compute_orderflow_liquidity(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
    provider = get_order_flow_provider()
    flow = provider.get_order_flow(ohlcv)
    amihud_scale = float(params.get("amihud_scale", 1e6))
    window = int(params.get("window", 20))
    return compute_liquidity_signal(ohlcv, flow, amihud_scale=amihud_scale, window=window)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rsi": {
        "display_name": "RSI",
        "description":  "Relative Strength Index. Oversold → buy, overbought → sell.",
        "type":         "A",
        "compute":      _compute_rsi,
        "params": {
            "period": {"label": "Period", "default": 14, "min": 2, "max": 50, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (5, 30)},
    },
    "macd": {
        "display_name": "MACD",
        "description":  "MACD histogram momentum. Bullish crossover → buy.",
        "type":         "B",
        "compute":      _compute_macd,
        "params": {
            "fast":   {"label": "Fast EMA",   "default": 12, "min": 2,  "max": 50,  "step": 1, "type": "int"},
            "slow":   {"label": "Slow EMA",   "default": 26, "min": 10, "max": 100, "step": 1, "type": "int"},
            "signal": {"label": "Signal EMA", "default": 9,  "min": 2,  "max": 30,  "step": 1, "type": "int"},
        },
        "tune_ranges": {"fast": (5, 20), "slow": (15, 50), "signal": (5, 15)},
    },
    "bollinger": {
        "display_name": "Bollinger Bands",
        "description":  "Price position within Bollinger Bands. Below lower → buy.",
        "type":         "A",
        "compute":      _compute_bollinger,
        "params": {
            "period":  {"label": "Period",   "default": 20, "min": 5,  "max": 100, "step": 1,   "type": "int"},
            "num_std": {"label": "Std Dev",  "default": 2.0,"min": 0.5,"max": 4.0, "step": 0.5, "type": "float"},
        },
        "tune_ranges": {"period": (10, 40), "num_std": (1.5, 3.0)},
    },
    "stochastic": {
        "display_name": "Stochastic",
        "description":  "Stochastic %K/%D oscillator. Oversold → buy, overbought → sell.",
        "type":         "A",
        "compute":      _compute_stochastic,
        "params": {
            "k_period": {"label": "K Period", "default": 14, "min": 2, "max": 50, "step": 1, "type": "int"},
            "d_period": {"label": "D Period", "default": 3,  "min": 1, "max": 15, "step": 1, "type": "int"},
        },
        "tune_ranges": {"k_period": (5, 25), "d_period": (2, 7)},
    },
    "dual_sma": {
        "display_name": "Dual SMA Crossover",
        "description":  "Golden cross / death cross. Fast SMA above slow → buy.",
        "type":         "B",
        "compute":      _compute_dual_sma,
        "params": {
            "fast": {"label": "Fast SMA", "default": 10, "min": 2,  "max": 100, "step": 1, "type": "int"},
            "slow": {"label": "Slow SMA", "default": 50, "min": 10, "max": 300, "step": 5, "type": "int"},
        },
        "tune_ranges": {"fast": (5, 30), "slow": (20, 100)},
    },
    "ema_crossover": {
        "display_name": "EMA Crossover",
        "description":  "Fast/slow EMA crossover with momentum confirmation.",
        "type":         "B",
        "compute":      _compute_ema_crossover,
        "params": {
            "fast": {"label": "Fast EMA", "default": 9,  "min": 2,  "max": 50,  "step": 1, "type": "int"},
            "slow": {"label": "Slow EMA", "default": 21, "min": 5,  "max": 100, "step": 1, "type": "int"},
        },
        "tune_ranges": {"fast": (3, 20), "slow": (15, 50)},
    },
    "momentum": {
        "display_name": "Momentum",
        "description":  "Raw price momentum (close - close[n]) normalized.",
        "type":         "B",
        "compute":      _compute_momentum,
        "params": {
            "period": {"label": "Period", "default": 10, "min": 2, "max": 50, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (5, 30)},
    },
    "roc": {
        "display_name": "Rate of Change",
        "description":  "Percentage rate of change, normalized via z-score.",
        "type":         "B",
        "compute":      _compute_roc,
        "params": {
            "period": {"label": "Period", "default": 10, "min": 2, "max": 50, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (5, 25)},
    },
    "atr_ratio": {
        "display_name": "ATR Ratio",
        "description":  "ATR relative to its MA — volatility expansion/contraction signal.",
        "type":         "B",
        "compute":      _compute_atr_ratio,
        "params": {
            "atr_period": {"label": "ATR Period", "default": 14, "min": 5, "max": 30, "step": 1, "type": "int"},
            "ma_period":  {"label": "MA Period",  "default": 50, "min": 20, "max": 200, "step": 5, "type": "int"},
        },
        "tune_ranges": {"atr_period": (7, 21), "ma_period": (20, 100)},
    },
    "vwap_dev": {
        "display_name": "VWAP Deviation",
        "description":  "Deviation from rolling VWAP. Below VWAP → mean-reversion buy.",
        "type":         "D",
        "compute":      _compute_vwap_dev,
        "params": {
            "period": {"label": "Period", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (10, 50)},
    },
    "cmf": {
        "display_name": "Chaikin Money Flow",
        "description":  "Accumulation/distribution signal. Positive CMF → buy.",
        "type":         "A",
        "compute":      _compute_cmf,
        "params": {
            "period": {"label": "Period", "default": 20, "min": 5, "max": 60, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (10, 40)},
    },
    "adx": {
        "display_name": "ADX / DI",
        "description":  "DI+/DI- direction × ADX strength. Strong uptrend → buy.",
        "type":         "B",
        "compute":      _compute_adx,
        "params": {
            "period": {"label": "Period", "default": 14, "min": 5, "max": 50, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (7, 28)},
    },
    "wyckoff": {
        "display_name": "Wyckoff",
        "description":  "Volume-price divergence. Accumulation → buy, distribution → sell.",
        "type":         "C",
        "compute":      _compute_wyckoff,
        "params": {
            "lookback": {"label": "Lookback", "default": 30, "min": 10, "max": 120, "step": 5, "type": "int"},
        },
        "tune_ranges": {"lookback": (15, 60)},
    },
    "range_pos": {
        "display_name": "Range Position",
        "description":  "Price position within rolling range. Mean-reversion signal.",
        "type":         "A",
        "compute":      _compute_range_pos,
        "params": {
            "period": {"label": "Period", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
        },
        "tune_ranges": {"period": (10, 50)},
    },

    "orderflow_vwap": {
        "display_name": "Order Flow VWAP",
        "description": "VWAP deviation normalized by ATR.",
        "type": "orderflow",
        "compute": _compute_orderflow_vwap,
        "params": {
            "atr_period": {"label": "ATR Period", "default": 14, "min": 5, "max": 50, "step": 1, "type": "int"},
            "clip_value": {"label": "Clip", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5, "type": "float"},
        },
        "tune_ranges": {"atr_period": (7, 28), "clip_value": (1.0, 3.0)},
    },
    "volume_profile": {
        "display_name": "Volume Profile",
        "description": "Rolling point-of-control proximity signal.",
        "type": "orderflow",
        "compute": _compute_orderflow_volume_profile,
        "params": {
            "window": {"label": "Window", "default": 20, "min": 5, "max": 120, "step": 1, "type": "int"},
            "bins": {"label": "Bins", "default": 16, "min": 4, "max": 40, "step": 1, "type": "int"},
            "near_poc_threshold": {"label": "Near POC %", "default": 0.002, "min": 0.0005, "max": 0.02, "step": 0.0005, "type": "float"},
        },
        "tune_ranges": {"window": (10, 60), "bins": (8, 24), "near_poc_threshold": (0.001, 0.01)},
    },
    "cumulative_delta": {
        "display_name": "Cumulative Delta",
        "description": "Rolling buy vs sell pressure from estimated order flow.",
        "type": "orderflow",
        "compute": _compute_orderflow_cumulative_delta,
        "params": {
            "window": {"label": "Window", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
            "clip_value": {"label": "Clip", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.1, "type": "float"},
        },
        "tune_ranges": {"window": (10, 50), "clip_value": (0.5, 2.0)},
    },
    "liquidity": {
        "display_name": "Liquidity Metrics",
        "description": "Spread/Amihud-based liquidity signal.",
        "type": "orderflow",
        "compute": _compute_orderflow_liquidity,
        "params": {
            "window": {"label": "Window", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
            "amihud_scale": {"label": "Amihud Scale", "default": 1000000.0, "min": 1000.0, "max": 10000000.0, "step": 1000.0, "type": "float"},
        },
        "tune_ranges": {"window": (10, 50), "amihud_scale": (1e4, 1e7)},
    },
    "mft_signal": {
        "display_name": "MFT Signal",
        "description":  "Simplified Market Field Theory gradient-direction signal.",
        "type":         "MFT",
        "compute":      _compute_mft_signal,
        "params": {
            "kernel": {"label": "Kernel", "default": "gaussian", "options": ["gaussian", "exp", "linear"], "type": "select"},
            "sigma": {"label": "Sigma", "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0, "type": "float"},
            "threshold": {"label": "Gradient Threshold", "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05, "type": "float"},
            "smooth_window": {"label": "Signal Smoothing", "default": 1, "min": 1, "max": 50, "step": 1, "type": "int"},
        },
        "tune_ranges": {"sigma": (3.0, 20.0), "threshold": (0.0, 1.0), "smooth_window": (1, 10)},
    },
    "mft_energy": {
        "display_name": "MFT Energy",
        "description":  "Relative field-energy signal (low activity bullish, high activity defensive).",
        "type":         "MFT",
        "compute":      _compute_mft_energy,
        "params": {
            "kernel": {"label": "Kernel", "default": "gaussian", "options": ["gaussian", "exp", "linear"], "type": "select"},
            "sigma": {"label": "Sigma", "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0, "type": "float"},
            "energy_window": {"label": "Energy Window", "default": 20, "min": 3, "max": 120, "step": 1, "type": "int"},
        },
        "tune_ranges": {"sigma": (3.0, 20.0), "energy_window": (10, 60)},
    },

    "phi_mft": {
        "display_name": "Phi-Bot (MFT)",
        "description":  "Backward-compatible alias for simplified MFT signal.",
        "type":         "MFT",
        "compute":      _compute_mft_signal,
        "params": {},
        "tune_ranges": {},
    },
}


def list_indicators() -> List[str]:
    return list(INDICATOR_REGISTRY.keys())


def get_indicator(name: str) -> Optional[Dict[str, Any]]:
    return INDICATOR_REGISTRY.get(name)


def compute_signal(name: str, ohlcv: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """Compute a normalized [-1, +1] signal for the given indicator."""
    info = INDICATOR_REGISTRY.get(name)
    if info is None:
        raise ValueError(f"Unknown indicator: '{name}'")
    p = {}
    for k, spec in info["params"].items():
        p[k] = spec["default"]
    if params:
        p.update(params)
    return info["compute"](ohlcv, p).rename(name)


def compute_all_signals(
    names: List[str],
    ohlcv: pd.DataFrame,
    params_map: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Compute multiple indicator signals, return as DataFrame."""
    signals = {}
    for name in names:
        p = (params_map or {}).get(name, {})
        try:
            signals[name] = compute_signal(name, ohlcv, p)
        except Exception as e:
            signals[name] = pd.Series(0.0, index=ohlcv.index, name=name)
    return pd.DataFrame(signals)
