"""Example strategy showing how to consume PostgreSQL options vendor data."""

from __future__ import annotations

import pandas as pd

from phi.data.cache import fetch_and_cache
from phi.options.data_adapter import adapt_for_backtesting


class OptionsExampleStrategy:
    """Tiny options signal example for integration verification."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def generate_signals(self, start: str, end: str) -> pd.DataFrame:
        raw = fetch_and_cache(vendor="postgres", symbol=self.symbol, start=start, end=end, timeframe="1m")
        df = adapt_for_backtesting(raw)

        if "open_interest" not in df.columns:
            df["open_interest"] = 0.0
        if "delta" not in df.columns:
            df["delta"] = 0.0
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["signal"] = 0
        df.loc[(df["delta"] > 0.5) & (df["volume"] > 0), "signal"] = 1
        return df[["signal", "delta", "open_interest"]]
