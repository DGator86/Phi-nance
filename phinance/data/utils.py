"""
phinance.data.utils
===================

Data cleaning, resampling, and normalisation helpers.

Functions
---------
  resample_ohlcv(df, rule)       — Resample OHLCV to a coarser timeframe
  fill_gaps(df, method)          — Forward-fill / interpolate missing bars
  compute_returns(df, col)       — Compute simple or log returns
  validate_date_range(start, end) — Raise if end < start
  clip_to_date_range(df, s, e)   — Slice DataFrame to [start, end]
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── Resampling ────────────────────────────────────────────────────────────────


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a coarser timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised OHLCV DataFrame with DatetimeIndex.
    rule : str
        Pandas offset alias (e.g. ``"4H"``, ``"1W"``, ``"ME"``).

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV with standard column names.

    Example
    -------
        daily_df = resample_ohlcv(intraday_df, "1D")
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    missing = [c for c in agg if c not in df.columns]
    if missing:
        raise ValueError(f"resample_ohlcv: missing columns {missing}")
    resampled = df.resample(rule).agg(agg).dropna(subset=["close"])
    logger.debug("Resampled %d → %d bars (rule=%s)", len(df), len(resampled), rule)
    return resampled


# ── Gap filling ───────────────────────────────────────────────────────────────


def fill_gaps(
    df: pd.DataFrame,
    method: Literal["ffill", "interpolate", "zero"] = "ffill",
) -> pd.DataFrame:
    """Fill missing / NaN values in an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    method : "ffill" | "interpolate" | "zero"

    Returns
    -------
    pd.DataFrame — same shape, gaps filled.
    """
    out = df.copy()
    if method == "ffill":
        out = out.ffill()
    elif method == "interpolate":
        out = out.interpolate(method="time")
    elif method == "zero":
        out = out.fillna(0.0)
    else:
        raise ValueError(f"Unknown fill method: {method!r}")
    return out


# ── Returns ───────────────────────────────────────────────────────────────────


def compute_returns(
    df: pd.DataFrame,
    col: str = "close",
    log: bool = False,
) -> pd.Series:
    """Compute simple or log returns for a price column.

    Parameters
    ----------
    df : pd.DataFrame
    col : str — column to compute returns on (default ``"close"``).
    log : bool — when True return natural log returns.

    Returns
    -------
    pd.Series
    """
    import numpy as np

    prices = df[col]
    if log:
        return pd.Series(
            np.log(prices / prices.shift(1)), index=df.index, name="log_return"
        )
    return prices.pct_change().rename("return")


# ── Date helpers ──────────────────────────────────────────────────────────────


def validate_date_range(start: str, end: str) -> None:
    """Raise ValueError if end < start.

    Parameters
    ----------
    start, end : str — ``"YYYY-MM-DD"``
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if e < s:
        raise ValueError(f"end date ({end}) must be >= start date ({start})")


def clip_to_date_range(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Slice a DatetimeIndex DataFrame to [start, end] inclusive.

    Parameters
    ----------
    df : pd.DataFrame
    start, end : str — ``"YYYY-MM-DD"``

    Returns
    -------
    pd.DataFrame
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return df.loc[(df.index >= s) & (df.index <= e)]
