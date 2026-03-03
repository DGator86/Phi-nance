"""
phinance.blending.regime_detector
==================================

Lightweight market-regime classifier that operates directly on OHLCV data —
no external regime engine required.

Returns a pd.Series of string regime labels aligned to the input index.

Regime taxonomy
---------------
  TREND_UP    — strong upward momentum
  TREND_DN    — strong downward momentum
  BREAKOUT_UP — price closes above rolling high channel
  BREAKOUT_DN — price closes below rolling low channel
  HIGHVOL     — ATR elevated vs its own historical mean
  LOWVOL      — ATR suppressed vs its own historical mean
  RANGE       — none of the above (default)

Algorithm priority (later layers override earlier):
  1. Volatility base (HIGHVOL / LOWVOL)
  2. Trend override  (TREND_UP / TREND_DN)
  3. Breakout        (BREAKOUT_UP / BREAKOUT_DN) — highest priority

Usage
-----
    from phinance.blending.regime_detector import detect_regime
    labels = detect_regime(ohlcv_df)  # pd.Series of strings
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_regime(ohlcv: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Classify market regime for each bar in *ohlcv*.

    Parameters
    ----------
    ohlcv    : pd.DataFrame — OHLCV with DatetimeIndex
    lookback : int          — rolling window in bars (default 20)

    Returns
    -------
    pd.Series — string regime labels aligned to ``ohlcv.index``
    """
    if ohlcv is None or len(ohlcv) < lookback * 2:
        idx = ohlcv.index if ohlcv is not None else pd.Index([])
        return pd.Series("RANGE", index=idx, dtype="object")

    close = ohlcv["close"]
    high  = ohlcv["high"]
    low   = ohlcv["low"]

    # True Range → ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback).mean()
    atr_pct = (atr / close.replace(0, np.nan)).fillna(0)

    # Trend: rate-of-change over lookback bars
    roc = close.pct_change(lookback).fillna(0)

    # Volatility regime: compare ATR% to its longer-window mean
    hist_window = min(60, max(lookback * 2, len(ohlcv) // 2))
    atr_ma = (
        atr_pct.rolling(hist_window)
        .mean()
        .fillna(atr_pct.expanding().mean())
    )
    vol_ratio = (atr_pct / atr_ma.replace(0, 1)).fillna(1.0)

    # Simplified ADX: directional movement
    dm_pos = (high - high.shift(1)).clip(lower=0).fillna(0)
    dm_neg = (low.shift(1) - low).clip(lower=0).fillna(0)
    atr_sum = tr.rolling(lookback).sum().replace(0, 1)
    di_pos  = dm_pos.rolling(lookback).sum() / atr_sum
    di_neg  = dm_neg.rolling(lookback).sum() / atr_sum
    di_sum  = (di_pos + di_neg).replace(0, 1)
    adx_proxy = ((di_pos - di_neg).abs() / di_sum).rolling(lookback).mean().fillna(0)

    # Channel breakout
    upper_ch = high.rolling(lookback).max().shift(1)
    lower_ch = low.rolling(lookback).min().shift(1)

    # Layer 1 — Volatility base
    regime = pd.Series("RANGE", index=close.index, dtype="object")
    regime[vol_ratio >= 1.5] = "HIGHVOL"
    regime[vol_ratio <= 0.55] = "LOWVOL"

    # Layer 2 — Trend override
    trend_strong = adx_proxy >= 0.22
    regime[trend_strong & (roc >= 0.02)]  = "TREND_UP"
    regime[trend_strong & (roc <= -0.02)] = "TREND_DN"

    # Layer 3 — Breakout override (highest priority)
    broke_up = upper_ch.notna() & (close >= upper_ch)
    broke_dn = lower_ch.notna() & (close <= lower_ch)
    regime[broke_up] = "BREAKOUT_UP"
    regime[broke_dn] = "BREAKOUT_DN"

    return regime.fillna("RANGE")


def regime_to_probs(labels: pd.Series) -> pd.DataFrame:
    """Convert a discrete regime label Series to a probability DataFrame.

    Each bar has probability 1.0 for its label and 0.0 for all others.
    This allows the ``regime_weighted`` blend method to accept output from
    ``detect_regime()`` directly (in addition to soft probability output
    from a full ML regime engine).

    Parameters
    ----------
    labels : pd.Series — string regime labels

    Returns
    -------
    pd.DataFrame — one-hot encoded probability DataFrame
    """
    all_regimes = [
        "TREND_UP", "TREND_DN", "RANGE", "HIGHVOL", "LOWVOL",
        "BREAKOUT_UP", "BREAKOUT_DN", "EXHAUST_REV",
    ]
    df = pd.DataFrame(0.0, index=labels.index, columns=all_regimes)
    for regime in all_regimes:
        df.loc[labels == regime, regime] = 1.0
    return df
