"""Feature engineering helpers for regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_ohlcv_columns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names to lowercase canonical names.

    Args:
        ohlcv: Raw OHLCV DataFrame.

    Returns:
        A copy of ``ohlcv`` with canonical column names.

    Raises:
        ValueError: If required OHLCV columns are missing.
    """
    cols = {c.lower(): c for c in ohlcv.columns}
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}")
    return ohlcv.rename(columns={cols[c]: c for c in required}).copy()


def extract_features(ohlcv: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Build a robust feature matrix from OHLCV bars.

    Features include simple return, log return, rolling volatility, ATR ratio,
    and log volume change.

    Args:
        ohlcv: Input bars with OHLCV columns.
        window: Rolling window used for volatility and ATR statistics.

    Returns:
        Feature DataFrame indexed by input timestamps with NaN/inf rows removed.

    Raises:
        ValueError: If ``window < 2`` or no usable feature rows remain.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    df = _normalize_ohlcv_columns(ohlcv)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=1e-8)

    returns = close.pct_change()
    log_returns = np.log(close / close.shift(1))
    rolling_vol = log_returns.rolling(window=window, min_periods=max(2, window // 2)).std()

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=max(2, window // 2)).mean()
    atr_ratio = atr / close.replace(0, np.nan)

    volume_change = np.log(volume / volume.shift(1))

    features = pd.DataFrame(
        {
            "returns": returns,
            "log_returns": log_returns,
            "rolling_vol": rolling_vol,
            "atr_ratio": atr_ratio,
            "volume_change": volume_change,
        },
        index=df.index,
    )

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Feature extraction produced no usable rows; check input data length/quality")
    return features
