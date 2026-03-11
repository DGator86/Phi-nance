"""Minimal options strategy for the new options backtest loop."""

from __future__ import annotations


class BasicOptionsStrategy:
    def __init__(self, symbol: str, threshold_delta: float = 0.5):
        self.symbol = symbol
        self.threshold_delta = threshold_delta

    def generate_signals(self, current_date, options_df, underlying_price):
        """Generate buy signals for liquid calls with high delta."""
        signals = []
        if options_df is None or options_df.empty:
            return signals

        if "optiontype" in options_df.columns and "option_type" not in options_df.columns:
            options_df = options_df.rename(columns={"optiontype": "option_type"})

        calls = options_df[options_df["option_type"].astype(str).str.upper().eq("CALL")]
        for _, row in calls.iterrows():
            if float(row.get("delta", 0.0)) > self.threshold_delta and float(row.get("volume", 0.0)) > 0:
                signals.append(
                    {
                        "action": "BUY",
                        "symbol": row.get("symbol", self.symbol),
                        "option_type": "CALL",
                        "strike": float(row["strike"]),
                        "expiration": row["expiration"],
                        "quantity": 1,
                        "price": float((row.get("bid", 0.0) + row.get("ask", 0.0)) / 2),
                        "delta": row.get("delta"),
                        "gamma": row.get("gamma"),
                        "theta": row.get("theta"),
                        "vega": row.get("vega"),
                        "rho": row.get("rho"),
                    }
                )
        return signals
