"""
phinance.strategies.ml_features
================================

Feature engineering for the ML classifier indicator.

Computes a rich set of technical features from OHLCV data:
  • Price-based: returns (1/5/20-day), log-return, gap
  • Rolling statistics: std, skew, min/max z-score
  • Trend: EMA ratios (fast/slow), MACD components
  • Momentum: RSI, ROC (5/20), MFI
  • Volatility: ATR ratio, Bollinger %B, realised vol
  • Volume: OBV trend, VWAP deviation, volume z-score
  • Regime: ADX proxy, trending flag

All features are computed without look-ahead bias (using only data
available at each bar).  NaNs are forward-filled then zero-filled.

Public API
----------
  build_features(ohlcv, window=20) → pd.DataFrame
  TARGET_COLUMN = "target"
  build_labels(ohlcv, horizon=1, threshold=0.0) → pd.Series
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "target"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs    = gain / loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat(
        [high - low, (high - pc).abs(), (low - pc).abs()], axis=1
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(span=2 * period - 1, adjust=False, min_periods=period).mean()


# ── Main feature builder ──────────────────────────────────────────────────────


def build_features(ohlcv: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Build ML feature matrix from OHLCV data.

    Parameters
    ----------
    ohlcv  : pd.DataFrame — OHLCV with columns [open, high, low, close, volume]
    window : int          — primary rolling window (default 20)

    Returns
    -------
    pd.DataFrame — feature matrix aligned to ohlcv.index; no look-ahead bias;
                   NaN rows at the start are zero-filled.
    """
    high   = ohlcv["high"].astype(float)
    low    = ohlcv["low"].astype(float)
    close  = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float)
    open_  = ohlcv["open"].astype(float)

    feats: dict = {}

    # ── Price returns ──────────────────────────────────────────────────────────
    feats["ret_1"]  = close.pct_change(1)
    feats["ret_5"]  = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["log_ret_1"] = np.log(close / close.shift(1))
    feats["gap"]    = (open_ - close.shift(1)) / close.shift(1)   # overnight gap

    # ── Rolling statistics ─────────────────────────────────────────────────────
    roll = close.pct_change(1).rolling(window)
    feats["vol_20"]    = roll.std()          # realised volatility proxy
    feats["skew_20"]   = roll.skew()

    roll_c = close.rolling(window)
    mu = roll_c.mean()
    sd = roll_c.std()
    feats["zscore_20"] = (close - mu) / sd.replace(0, np.nan)
    feats["dist_hi"]   = (close - high.rolling(window).max()) / close   # distance from recent high
    feats["dist_lo"]   = (close - low.rolling(window).min()) / close    # distance from recent low

    # ── Trend: EMA ratios ──────────────────────────────────────────────────────
    ema_fast = _ema(close, 8)
    ema_slow = _ema(close, 21)
    ema_200  = _ema(close, 200)
    feats["ema_ratio_8_21"]   = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    feats["ema_ratio_21_200"] = (ema_slow - ema_200) / ema_200.replace(0, np.nan)
    feats["price_vs_ema21"]   = (close - ema_slow) / ema_slow.replace(0, np.nan)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line   = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd_line, 9)
    feats["macd_hist"] = macd_line - macd_signal
    feats["macd_line"] = macd_line

    # ── Momentum ──────────────────────────────────────────────────────────────
    feats["rsi_14"]  = _rsi(close, 14)
    feats["rsi_7"]   = _rsi(close, 7)
    feats["roc_5"]   = (close / close.shift(5) - 1) * 100
    feats["roc_20"]  = (close / close.shift(20) - 1) * 100

    # Money Flow Index (simplified)
    typical = (high + low + close) / 3.0
    mf_raw  = typical * volume
    up_mf   = mf_raw.where(typical > typical.shift(1), 0.0).rolling(14).sum()
    dn_mf   = mf_raw.where(typical < typical.shift(1), 0.0).rolling(14).sum()
    mfr     = up_mf / dn_mf.replace(0, np.nan)
    feats["mfi_14"] = (100 - 100 / (1 + mfr)).fillna(50.0)

    # ── Volatility / Bands ────────────────────────────────────────────────────
    atr = _atr(high, low, close, 14)
    feats["atr_ratio"] = atr / close.replace(0, np.nan)   # ATR as % of price

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    feats["bb_pct_b"] = (close - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    feats["vol_ratio"] = volume / vol_ma.replace(0, np.nan)   # relative volume
    vol_std = volume.rolling(20).std()
    feats["vol_zscore"] = (volume - vol_ma) / vol_std.replace(0, np.nan)

    # OBV trend (sign of OBV change over window)
    obv = (np.sign(close.diff()) * volume).cumsum()
    feats["obv_trend"] = (obv - obv.rolling(window).mean()) / obv.rolling(window).std().replace(0, np.nan)

    # VWAP deviation
    typical_pv = typical * volume
    cum_tpv = typical_pv.rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    feats["vwap_dev"] = (close - vwap) / vwap.replace(0, np.nan)

    # ── Regime / ADX proxy ────────────────────────────────────────────────────
    # ADX proxy: abs(EMA ratio) as trend strength
    feats["adx_proxy"]    = feats["ema_ratio_8_21"].abs()
    feats["trending_flag"] = (feats["adx_proxy"] > 0.01).astype(float)

    # ── Bar shape ─────────────────────────────────────────────────────────────
    candle_range = (high - low).replace(0, np.nan)
    feats["body_ratio"] = (close - open_).abs() / candle_range   # body / range
    feats["upper_wick"]  = (high - pd.concat([close, open_], axis=1).max(axis=1)) / candle_range
    feats["lower_wick"]  = (pd.concat([close, open_], axis=1).min(axis=1) - low) / candle_range

    df = pd.DataFrame(feats, index=ohlcv.index)
    return df.ffill().fillna(0.0)


def build_labels(
    ohlcv: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Build directional labels for supervised learning.

    Label at bar t = 1 if close[t+horizon] > close[t] * (1+threshold), else 0.

    Parameters
    ----------
    ohlcv     : pd.DataFrame — OHLCV
    horizon   : int          — forward-return horizon in bars (default 1)
    threshold : float        — minimum return magnitude to label as 1 (default 0)

    Returns
    -------
    pd.Series[int] — binary label (1=up, 0=down), NaN at last ``horizon`` bars
    """
    close = ohlcv["close"].astype(float)
    fwd_return = close.shift(-horizon) / close - 1.0
    labels = (fwd_return > threshold).astype(float)
    labels.iloc[-horizon:] = np.nan
    labels.name = TARGET_COLUMN
    return labels
