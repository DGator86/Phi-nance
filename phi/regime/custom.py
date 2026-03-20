"""Convenience wrappers for symbol/date oriented regime detection."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import pandas as pd

from phi.data.fetchers import fetch
from phi.options.data_adapter import fetch_options_data
from phi.regime.regime_definitions import compose_detailed_regime, infer_base_regime_from_prices, infer_volatility_regime


def get_detailed_regime_for_symbol(
    symbol: str,
    as_of: str | date | None = None,
    lookback_days: int = 252,
    vendor: str = "yfinance",
    vol_window: int = 20,
) -> str:
    """Fetch OHLCV and return the latest detailed regime label for ``symbol``."""
    end_dt = pd.Timestamp(as_of).date() if as_of is not None else pd.Timestamp.utcnow().date()
    start_dt = end_dt - timedelta(days=max(lookback_days, vol_window + 10))
    ohlcv = fetch(symbol=symbol, start=start_dt, end=end_dt + timedelta(days=1), timeframe="1D", vendor=vendor)
    if ohlcv is None or ohlcv.empty:
        return "RANGING_NORMAL_VOL"
    vol_regime = infer_volatility_regime(ohlcv=ohlcv, window=vol_window)
    base = infer_base_regime_from_prices(ohlcv)
    return compose_detailed_regime(base_regime=base, volatility_regime=vol_regime).label


def get_iv_regime(
    symbol: str,
    as_of: str | date,
    low_threshold: float = 20.0,
    high_threshold: float = 30.0,
    base_regime_fn: Callable[[pd.DataFrame], str] | None = None,
) -> str:
    """Compose a detailed regime using options implied vol instead of realized volatility."""
    as_of_date = pd.Timestamp(as_of).date()
    options_df = fetch_options_data(symbol, str(as_of_date), str(as_of_date))
    if options_df is None or options_df.empty or "volatility" not in options_df.columns:
        return "UNKNOWN"

    avg_iv = float(options_df["volatility"].dropna().mean())
    if avg_iv < low_threshold:
        vol_regime = "LOW_VOL"
    elif avg_iv > high_threshold:
        vol_regime = "HIGH_VOL"
    else:
        vol_regime = "NORMAL_VOL"

    end_dt = as_of_date
    start_dt = end_dt - timedelta(days=252)
    ohlcv = fetch(symbol=symbol, start=start_dt, end=end_dt + timedelta(days=1), timeframe="1D", vendor="yfinance")
    base = "RANGING"
    if ohlcv is not None and not ohlcv.empty:
        base = base_regime_fn(ohlcv) if base_regime_fn is not None else infer_base_regime_from_prices(ohlcv)

    return compose_detailed_regime(base_regime=base, volatility_regime=vol_regime).label
