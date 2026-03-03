"""
phinance.data.features
========================

Technical Feature Engineering — extract a rich feature matrix from OHLCV data.

All features are normalised or bounded so they can be fed directly into ML
models (sklearn, XGBoost, PyTorch, etc.) without additional preprocessing.

Feature groups
--------------
* **Price momentum**     — returns over 1, 3, 5, 10, 20 bars
* **Trend**              — SMA/EMA crosses, price vs moving average
* **Volatility**         — ATR-normalised range, Bollinger width, std of returns
* **Volume**             — volume ratio, OBV slope, volume z-score
* **Oscillators**        — RSI, Stochastic %D, Williams %R, CCI (all bounded [0,1])
* **Regime**             — volatility regime, trend strength (ADX-like)
* **Calendar**           — day-of-week, month, quarter dummies (optional)

Usage
-----
    from phinance.data.features import build_feature_matrix

    X = build_feature_matrix(ohlcv_df, include_calendar=True)
    # X is a pd.DataFrame with DatetimeIndex aligned to ohlcv_df
    # First `warmup` rows will contain NaN — drop them with X.dropna()
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from phinance.utils.logging import get_logger

_log = get_logger(__name__)

# Warmup bars required before features become valid
FEATURE_WARMUP_BARS: int = 50


# ── Main entry point ──────────────────────────────────────────────────────────

def build_feature_matrix(
    ohlcv:            pd.DataFrame,
    include_calendar: bool = False,
    price_lags:       Optional[List[int]] = None,
    sma_windows:      Optional[List[int]] = None,
    ema_windows:      Optional[List[int]] = None,
    rsi_period:       int = 14,
    atr_period:       int = 14,
    bb_period:        int = 20,
    vol_period:       int = 20,
) -> pd.DataFrame:
    """Build a normalised technical feature matrix from OHLCV data.

    Parameters
    ----------
    ohlcv            : pd.DataFrame — columns ``[open, high, low, close, volume]``
    include_calendar : bool         — add day-of-week / month / quarter dummies
    price_lags       : list[int]    — return lags in bars (default [1,3,5,10,20])
    sma_windows      : list[int]    — SMA windows     (default [10, 20, 50])
    ema_windows      : list[int]    — EMA windows     (default [12, 26])
    rsi_period       : int          — RSI look-back
    atr_period       : int          — ATR look-back
    bb_period        : int          — Bollinger Band look-back
    vol_period       : int          — volatility / z-score look-back

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same DatetimeIndex as *ohlcv*.
        NaN in the first ``FEATURE_WARMUP_BARS`` rows.
    """
    price_lags  = price_lags  or [1, 3, 5, 10, 20]
    sma_windows = sma_windows or [10, 20, 50]
    ema_windows = ema_windows or [12, 26]

    close  = ohlcv["close"].astype(float)
    high   = ohlcv["high"].astype(float)
    low    = ohlcv["low"].astype(float)
    open_  = ohlcv["open"].astype(float)
    volume = ohlcv["volume"].astype(float)

    features: dict[str, pd.Series] = {}

    # ── 1. Price momentum (log returns) ──────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    for lag in price_lags:
        features[f"ret_{lag}b"] = (close / close.shift(lag) - 1.0).clip(-0.5, 0.5)

    # ── 2. Trend: price vs SMAs ───────────────────────────────────────────────
    for w in sma_windows:
        sma = close.rolling(window=w, min_periods=w).mean()
        features[f"price_vs_sma{w}"] = ((close - sma) / sma.replace(0, np.nan)).clip(-0.5, 0.5)

    # ── 3. SMA cross (fast/slow) ──────────────────────────────────────────────
    if len(sma_windows) >= 2:
        s_fast = close.rolling(sma_windows[0], min_periods=sma_windows[0]).mean()
        s_slow = close.rolling(sma_windows[1], min_periods=sma_windows[1]).mean()
        features["sma_cross"] = ((s_fast - s_slow) / s_slow.replace(0, np.nan)).clip(-0.5, 0.5)

    # ── 4. EMA trend ──────────────────────────────────────────────────────────
    for w in ema_windows:
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
        features[f"price_vs_ema{w}"] = ((close - ema) / ema.replace(0, np.nan)).clip(-0.5, 0.5)

    if len(ema_windows) >= 2:
        e_fast = close.ewm(span=ema_windows[0], adjust=False, min_periods=ema_windows[0]).mean()
        e_slow = close.ewm(span=ema_windows[1], adjust=False, min_periods=ema_windows[1]).mean()
        macd_line = e_fast - e_slow
        signal_ln = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        features["macd_hist"] = ((macd_line - signal_ln) / close.replace(0, np.nan)).clip(-0.05, 0.05) * 20

    # ── 5. Volatility ─────────────────────────────────────────────────────────
    # True Range normalised by close
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    features["atr_norm"] = (atr / close.replace(0, np.nan)).clip(0, 0.2)

    # Bollinger Band width (% of midline)
    sma_bb = close.rolling(bb_period, min_periods=bb_period).mean()
    std_bb = close.rolling(bb_period, min_periods=bb_period).std(ddof=1)
    features["bb_width"] = (2 * 2.0 * std_bb / sma_bb.replace(0, np.nan)).clip(0, 0.5)

    # Bollinger %B  (0 = lower band, 1 = upper band)
    upper_bb = sma_bb + 2.0 * std_bb
    lower_bb = sma_bb - 2.0 * std_bb
    bb_rng   = (upper_bb - lower_bb).replace(0, np.nan)
    features["bb_pct_b"] = ((close - lower_bb) / bb_rng).clip(0.0, 1.0)

    # Rolling return std (realised vol)
    features["ret_std"] = log_ret.rolling(vol_period, min_periods=vol_period).std()

    # ── 6. Volume features ────────────────────────────────────────────────────
    vol_ma = volume.rolling(vol_period, min_periods=vol_period).mean()
    features["volume_ratio"] = (volume / vol_ma.replace(0, np.nan)).clip(0, 5) / 5.0

    # OBV slope (normalised)
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    obv_prev = obv.shift(10)
    features["obv_slope"] = ((obv - obv_prev) / (obv_prev.abs().replace(0, np.nan))).clip(-1, 1)

    # Volume z-score
    vol_std = volume.rolling(vol_period, min_periods=vol_period).std(ddof=1)
    features["volume_zscore"] = ((volume - vol_ma) / vol_std.replace(0, np.nan)).clip(-3, 3) / 3.0

    # ── 7. Oscillators (bounded [0, 1]) ───────────────────────────────────────

    # RSI (Wilder SMMA)
    delta   = close.diff()
    avg_g   = delta.clip(lower=0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_l   = (-delta).clip(lower=0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs      = avg_g / avg_l.where(avg_l != 0, other=1e-10)
    rsi_raw = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    features["rsi"] = (rsi_raw / 100.0).clip(0, 1)   # [0,1]

    # Stochastic %D
    ll    = low.rolling(14, min_periods=14).min()
    hh    = high.rolling(14, min_periods=14).max()
    rng14 = (hh - ll).replace(0, np.nan)
    fast_k = ((close - ll) / rng14 * 100.0).clip(0, 100)
    pct_d  = fast_k.rolling(3, min_periods=1).mean().rolling(3, min_periods=3).mean()
    features["stoch_d"] = (pct_d / 100.0).clip(0, 1)

    # Williams %R → convert to [0, 1]  (−100 = oversold, 0 = overbought)
    features["williams_r"] = (1.0 + (pct_d - 100.0) / 100.0).clip(0, 1)

    # CCI: (close - sma) / (0.015 * mean_abs_deviation), scaled to [-1,1] → shift to [0,1]
    sma20  = close.rolling(20, min_periods=20).mean()
    mad    = close.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_raw = ((close - sma20) / (0.015 * mad.replace(0, np.nan))).clip(-3, 3) / 3.0
    features["cci"] = ((cci_raw + 1.0) / 2.0).clip(0, 1)

    # ── 8. Candle patterns ────────────────────────────────────────────────────
    body   = (close - open_).abs()
    candle = (high - low).replace(0, np.nan)
    features["body_ratio"]  = (body / candle).clip(0, 1)
    features["close_pct"]   = ((close - low) / candle).clip(0, 1)   # close position in range

    # ── 9. Calendar dummies (optional) ───────────────────────────────────────
    if include_calendar and hasattr(ohlcv.index, "day_of_week"):
        idx = ohlcv.index
        for d in range(5):
            features[f"dow_{d}"] = pd.Series(
                (idx.day_of_week == d).astype(float), index=idx
            )
        for m in range(1, 13):
            features[f"month_{m}"] = pd.Series(
                (idx.month == m).astype(float), index=idx
            )

    # ── Assemble ──────────────────────────────────────────────────────────────
    X = pd.DataFrame(features, index=ohlcv.index)
    _log.debug("build_feature_matrix: shape=%s", X.shape)
    return X


# ── Convenience helpers ───────────────────────────────────────────────────────

def drop_warmup(X: pd.DataFrame, warmup: int = FEATURE_WARMUP_BARS) -> pd.DataFrame:
    """Drop the first *warmup* rows (NaN warmup period) from a feature matrix."""
    return X.iloc[warmup:].copy()


def feature_names(
    price_lags:  Optional[List[int]] = None,
    sma_windows: Optional[List[int]] = None,
    ema_windows: Optional[List[int]] = None,
    include_calendar: bool = False,
) -> List[str]:
    """Return the list of feature column names without computing them."""
    price_lags  = price_lags  or [1, 3, 5, 10, 20]
    sma_windows = sma_windows or [10, 20, 50]
    ema_windows = ema_windows or [12, 26]
    names: List[str] = []
    names += [f"ret_{lag}b" for lag in price_lags]
    names += [f"price_vs_sma{w}" for w in sma_windows]
    if len(sma_windows) >= 2:
        names.append("sma_cross")
    names += [f"price_vs_ema{w}" for w in ema_windows]
    if len(ema_windows) >= 2:
        names.append("macd_hist")
    names += ["atr_norm", "bb_width", "bb_pct_b", "ret_std"]
    names += ["volume_ratio", "obv_slope", "volume_zscore"]
    names += ["rsi", "stoch_d", "williams_r", "cci"]
    names += ["body_ratio", "close_pct"]
    if include_calendar:
        names += [f"dow_{d}" for d in range(5)]
        names += [f"month_{m}" for m in range(1, 13)]
    return names
