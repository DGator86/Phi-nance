"""Regime-mapped options strategy classes for the lightweight backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    """Portable order/signal payload used by ``phi.backtest.engine.run_options_backtest``."""

    action: str
    symbol: str
    option_type: str
    strike: float
    expiration: Any
    quantity: int
    price: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "quantity": self.quantity,
            "price": self.price,
        }


class BaseOptionsStrategy(ABC):
    """Common interface for regime-aware strategy classes."""

    def __init__(self, symbol: str, min_volume: int = 1) -> None:
        self.symbol = symbol
        self.min_volume = min_volume

    @abstractmethod
    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        """Return order dictionaries that conform to the options backtest engine."""

    def get_params(self) -> dict[str, Any]:
        """Return strategy parameters for experiment tracking."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    @staticmethod
    def _normalized(options_df: pd.DataFrame) -> pd.DataFrame:
        if options_df is None or options_df.empty:
            return pd.DataFrame()
        df = options_df.copy()
        if "optiontype" in df.columns and "option_type" not in df.columns:
            df = df.rename(columns={"optiontype": "option_type"})
        if "option_type" in df.columns:
            df["option_type"] = df["option_type"].astype(str).str.upper()
        return df

    def _mid_price(self, row: pd.Series) -> float:
        return float((row.get("bid", 0.0) + row.get("ask", 0.0)) / 2)

    def _liquid(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        return df[volume >= float(self.min_volume)]

    def _order(self, action: str, row: pd.Series, quantity: int = 1) -> dict[str, Any]:
        return StrategySignal(
            action=action,
            symbol=str(row.get("symbol", self.symbol)),
            option_type=str(row.get("option_type", "CALL")).upper(),
            strike=float(row["strike"]),
            expiration=row["expiration"],
            quantity=quantity,
            price=self._mid_price(row),
        ).as_dict()


class _DeltaDirectionalStrategy(BaseOptionsStrategy):
    option_type: str = "CALL"
    action: str = "BUY"
    target_abs_delta: float = 0.35

    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        df = self._normalized(options_df)
        if df.empty:
            return []
        subset = self._liquid(df[df["option_type"].eq(self.option_type)])
        if subset.empty:
            return []
        subset = subset.copy()
        subset["delta_abs_diff"] = (subset.get("delta", 0.0).astype(float).abs() - self.target_abs_delta).abs()
        best = subset.sort_values(["delta_abs_diff", "expiration", "strike"]).iloc[0]
        return [self._order(self.action, best)]


class LongCallStrategy(_DeltaDirectionalStrategy):
    option_type = "CALL"
    action = "BUY"
    target_abs_delta = 0.50


class LongPutStrategy(_DeltaDirectionalStrategy):
    option_type = "PUT"
    action = "BUY"
    target_abs_delta = 0.50


class CashSecuredPutStrategy(_DeltaDirectionalStrategy):
    option_type = "PUT"
    action = "SELL"
    target_abs_delta = 0.25


class LongStraddleStrategy(BaseOptionsStrategy):
    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        df = self._normalized(options_df)
        if df.empty:
            return []
        liquid = self._liquid(df)
        if liquid.empty:
            return []
        liquid = liquid.copy()
        liquid["moneyness_diff"] = (liquid["strike"].astype(float) - float(underlying_price)).abs()
        atm = liquid.sort_values(["moneyness_diff", "expiration"]).head(1)
        if atm.empty:
            return []
        strike = float(atm.iloc[0]["strike"])
        expiry = atm.iloc[0]["expiration"]
        call = liquid[(liquid["option_type"].eq("CALL")) & (liquid["strike"].astype(float).eq(strike)) & (liquid["expiration"].eq(expiry))]
        put = liquid[(liquid["option_type"].eq("PUT")) & (liquid["strike"].astype(float).eq(strike)) & (liquid["expiration"].eq(expiry))]
        if call.empty or put.empty:
            return []
        return [self._order("BUY", call.iloc[0]), self._order("BUY", put.iloc[0])]


class LongStrangleStrategy(BaseOptionsStrategy):
    def __init__(self, symbol: str, min_volume: int = 1, wing_pct: float = 0.05) -> None:
        super().__init__(symbol=symbol, min_volume=min_volume)
        self.wing_pct = wing_pct

    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        df = self._normalized(options_df)
        if df.empty:
            return []
        liquid = self._liquid(df)
        low_target = float(underlying_price) * (1.0 - self.wing_pct)
        high_target = float(underlying_price) * (1.0 + self.wing_pct)
        puts = liquid[liquid["option_type"].eq("PUT")].copy()
        calls = liquid[liquid["option_type"].eq("CALL")].copy()
        if puts.empty or calls.empty:
            return []
        puts["strike_diff"] = (puts["strike"].astype(float) - low_target).abs()
        calls["strike_diff"] = (calls["strike"].astype(float) - high_target).abs()
        return [self._order("BUY", calls.sort_values("strike_diff").iloc[0]), self._order("BUY", puts.sort_values("strike_diff").iloc[0])]


class _VerticalSpread(BaseOptionsStrategy):
    direction: str = "bull"
    option_type: str = "CALL"
    short_first: bool = False
    width_pct: float = 0.05

    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        df = self._normalized(options_df)
        chain = self._liquid(df[df["option_type"].eq(self.option_type)]).copy()
        if chain.empty:
            return []
        chain["strike_diff"] = (chain["strike"].astype(float) - float(underlying_price)).abs()
        near = chain.sort_values(["strike_diff", "expiration"]).iloc[0]
        base_strike = float(near["strike"])
        target = base_strike * (1.0 + self.width_pct if self.direction == "bull" and self.option_type == "CALL" else 1.0 - self.width_pct)
        if self.direction == "bull" and self.option_type == "PUT":
            target = base_strike * (1.0 - self.width_pct)
        if self.direction == "bear" and self.option_type == "CALL":
            target = base_strike * (1.0 + self.width_pct)
        chain["target_diff"] = (chain["strike"].astype(float) - target).abs()
        hedge = chain.sort_values(["target_diff", "expiration"]).iloc[0]
        first_action = "SELL" if self.short_first else "BUY"
        second_action = "BUY" if self.short_first else "SELL"
        return [self._order(first_action, near), self._order(second_action, hedge)]


class BullCallSpreadStrategy(_VerticalSpread):
    direction = "bull"
    option_type = "CALL"
    short_first = False


class BearPutSpreadStrategy(_VerticalSpread):
    direction = "bear"
    option_type = "PUT"
    short_first = False


class BullPutSpreadStrategy(_VerticalSpread):
    direction = "bull"
    option_type = "PUT"
    short_first = True


class BearCallSpreadStrategy(_VerticalSpread):
    direction = "bear"
    option_type = "CALL"
    short_first = True


class IronCondorStrategy(LongStrangleStrategy):
    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        legs = super().generate_signals(current_date, options_df, underlying_price)
        if len(legs) != 2:
            return []
        short_call, short_put = legs[0], legs[1]
        short_call["action"] = "SELL"
        short_put["action"] = "SELL"
        return [short_call, short_put]


class IronButterflyStrategy(LongStraddleStrategy):
    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float) -> list[dict[str, Any]]:
        legs = super().generate_signals(current_date, options_df, underlying_price)
        if len(legs) != 2:
            return []
        for leg in legs:
            leg["action"] = "SELL"
        return legs


# Lightweight aliases so every approved strategy has a concrete class.
class LongCallButterflyStrategy(IronButterflyStrategy):
    pass


class LongPutButterflyStrategy(IronButterflyStrategy):
    pass


class CalendarSpreadStrategy(BullCallSpreadStrategy):
    pass


class DiagonalSpreadStrategy(BullCallSpreadStrategy):
    pass


STRATEGY_CLASS_MAP: dict[str, type[BaseOptionsStrategy]] = {
    "Long Call": LongCallStrategy,
    "Bull Call Spread": BullCallSpreadStrategy,
    "Bull Put Spread": BullPutSpreadStrategy,
    "Long Put": LongPutStrategy,
    "Bear Put Spread": BearPutSpreadStrategy,
    "Bear Call Spread": BearCallSpreadStrategy,
    "Long Straddle": LongStraddleStrategy,
    "Long Strangle": LongStrangleStrategy,
    "Iron Condor": IronCondorStrategy,
    "Iron Butterfly": IronButterflyStrategy,
    "Long Call Butterfly": LongCallButterflyStrategy,
    "Long Put Butterfly": LongPutButterflyStrategy,
    "Calendar Spread": CalendarSpreadStrategy,
    "Diagonal Spread": DiagonalSpreadStrategy,
    "Cash-Secured Put": CashSecuredPutStrategy,
}
