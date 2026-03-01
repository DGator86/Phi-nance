"""
Simple indicator signal computation from OHLCV.
Returns normalized signal series (-1 to 1 scale) for blending.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


def _normalize_signal(s: pd.Series) -> pd.Series:
    """Clip and scale to roughly [-1, 1]."""
    if s.isna().all():
        return s
    q = s.quantile([0.01, 0.99])
    lo, hi = q.iloc[0], q.iloc[1]
    r = hi - lo
    if r == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / r - 0.5) * 2


def compute_rsi(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI as normalized signal: oversold -> positive, overbought -> negative."""
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Normalize: 30 -> +1 (oversold, expect up), 70 -> -1 (overbought, expect down)
    signal = (50 - rsi) / 50
    return signal.clip(-1, 1)


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD histogram as normalized signal."""
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return _normalize_signal(hist)


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.Series:
    """Bollinger: below lower band -> positive, above upper -> negative."""
    close = df["close"]
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # Position within bands: -1 at upper, +1 at lower
    width = upper - lower
    width = width.replace(0, np.nan)
    pos = (close - lower) / width
    signal = (0.5 - pos) * 2
    return signal.clip(-1, 1)


def compute_dual_sma(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """Dual SMA: fast > slow -> positive, fast < slow -> negative."""
    close = df["close"]
    sma_fast = close.rolling(window=fast, min_periods=fast).mean()
    sma_slow = close.rolling(window=slow, min_periods=slow).mean()
    diff = (sma_fast - sma_slow) / sma_slow
    return _normalize_signal(diff)


def compute_mean_reversion(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Mean reversion: below SMA -> positive, above -> negative."""
    close = df["close"]
    sma = close.rolling(window=period, min_periods=period).mean()
    dev = (sma - close) / sma.replace(0, np.nan)
    return _normalize_signal(dev)


def compute_breakout(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Breakout: above channel -> positive, below -> negative."""
    high = df["high"]
    low = df["low"]
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    close = df["close"]
    mid = (upper + lower) / 2
    width = (upper - lower).replace(0, np.nan)
    pos = (close - mid) / width
    return _normalize_signal(pos)


def compute_buy_hold(df: pd.DataFrame) -> pd.Series:
    """Buy & hold: always +0.5 (slight bullish)."""
    return pd.Series(0.5, index=df.index)


def compute_vwap(df: pd.DataFrame, band_pct: float = 0.5) -> pd.Series:
    """VWAP deviation signal — optimised for intraday timeframes (1m – 1H).

    VWAP is computed as the session cumulative (typical_price * volume) /
    cumulative volume.  The signal measures how far price has stretched from
    VWAP and fades back toward it:

    * Price well **below** VWAP → signal approaches **+1** (mean-revert long)
    * Price well **above** VWAP → signal approaches **-1** (mean-revert short)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a datetime index.
    band_pct : float
        Half-band width as a percentage of VWAP (default 0.5 %).  A deviation
        of ``band_pct`` maps to ±1.  Tighten for scalping (0.2), widen for
        swing (1.0).

    Returns
    -------
    pd.Series
        Normalised signal in [-1, 1].

    Notes
    -----
    On daily data VWAP resets once per day, collapsing to the typical price of
    each bar — the indicator still works but is less meaningful.  Prefer
    intraday timeframes (1m, 5m, 15m, 1H) for best results.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)

    # Group by calendar date so VWAP resets each session.
    dates = df.index.normalize() if hasattr(df.index, "normalize") else pd.to_datetime(df.index).normalize()
    cum_tp_vol = (tp * vol).groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    vwap = cum_tp_vol / cum_vol

    dev_pct = (df["close"] - vwap) / vwap.replace(0, np.nan) * 100
    # Clamp: band_pct above → -1, band_pct below → +1
    signal = -(dev_pct / band_pct).clip(-1, 1)
    return signal.fillna(0.0)


INDICATOR_COMPUTERS: Dict[str, Callable[..., pd.Series]] = {
    "RSI": compute_rsi,
    "MACD": compute_macd,
    "Bollinger": compute_bollinger,
    "Dual SMA": compute_dual_sma,
    "Mean Reversion": compute_mean_reversion,
    "Breakout": compute_breakout,
    "Buy & Hold": compute_buy_hold,
    "VWAP": compute_vwap,
}


_PARAM_MAP = {
    "RSI": {"rsi_period": "period", "oversold": "oversold", "overbought": "overbought"},
    "MACD": {"fast_period": "fast", "slow_period": "slow", "signal_period": "signal_period"},
    "Bollinger": {"bb_period": "period", "num_std": "num_std"},
    "Dual SMA": {"fast_period": "fast", "slow_period": "slow"},
    "Mean Reversion": {"sma_period": "period"},
    "Breakout": {"channel_period": "period"},
    "VWAP": {"band_pct": "band_pct"},
}


def compute_indicator(name: str, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Compute indicator signal by name with params."""
    fn = INDICATOR_COMPUTERS.get(name)
    if fn is None:
        return pd.Series(0.0, index=df.index)
    pmap = _PARAM_MAP.get(name, {})
    kwargs = {pmap.get(k, k): v for k, v in params.items()}
    try:
        return fn(df, **kwargs)
    except Exception:
        return pd.Series(0.0, index=df.index)
