"""Simple options strategy scaffold for live-vendor testing."""

from __future__ import annotations

import pandas as pd

from phi.options.data_adapter import fetch_options_data


class MyOptionsStrategy:
    """Generate toy options signals from adapted postgres data."""

    def __init__(self, symbol: str, lookback_days: int = 30) -> None:
        self.symbol = symbol
        self.lookback_days = lookback_days

    def generate_signals(self, end_date: str) -> pd.DataFrame:
        end_ts = pd.Timestamp(end_date)
        start_ts = end_ts - pd.Timedelta(days=self.lookback_days)

        df = fetch_options_data(
            symbol=self.symbol,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
        )
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["signal"] = 0
        df.loc[(df["delta"] > 0.6) & (df["volume"] > 100), "signal"] = 1
        df.loc[df["delta"] < 0.3, "signal"] = -1

        return df[["signal", "delta", "volume", "open_interest"]]
