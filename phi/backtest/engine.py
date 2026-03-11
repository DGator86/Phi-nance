"""Backtest engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any, Dict

import pandas as pd

from phi.logging import get_logger
from phi.options.data_adapter import fetch_options_data
from phi.options.portfolio import Portfolio
from phi.options.position import OptionPosition

from phi.run_config import RunConfig

logger = get_logger(__name__)


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, config: RunConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute a backtest and return results (metrics + artifacts)."""


def run_portfolio_backtest_engine(config: RunConfig, data: dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compatibility wrapper for the portfolio-aware direct engine."""
    from phi.backtest.direct import run_portfolio_backtest

    return run_portfolio_backtest(
        data_dict=data,
        indicators=config.indicators,
        blend_weights=config.blend_weights,
        blend_method=config.blend_method,
        initial_capital=config.initial_capital,
        allocation_strategy=getattr(config, "allocation_strategy", "equal_weight"),
        allocation_params=getattr(config, "allocation_params", {}),
        rebalance_frequency=getattr(config, "rebalance_frequency", "M"),
        rebalance_threshold=getattr(config, "rebalance_threshold", None),
    )


def run_options_backtest(strategy, symbols, start_date, end_date, initial_cash, vendor: str = "yfinance") -> Portfolio:
    """Run a simple date-by-date options backtest loop."""
    from phi.data.fetchers import fetch

    if not symbols:
        raise ValueError("symbols must include at least one underlying")

    start = pd.Timestamp(start_date).date() if not isinstance(start_date, date) else start_date
    end = pd.Timestamp(end_date).date() if not isinstance(end_date, date) else end_date

    underlying_data = {
        symbol: fetch(symbol, start, end + timedelta(days=1), timeframe="1D", vendor=vendor)
        for symbol in symbols
    }

    portfolio = Portfolio(initial_cash)
    current_date = start
    while current_date <= end:
        underlying_prices: dict[str, float] = {}
        for symbol in symbols:
            frame = underlying_data.get(symbol)
            if frame is None or frame.empty:
                continue
            day_slice = frame.loc[frame.index.date == current_date]
            if not day_slice.empty:
                underlying_prices[symbol] = float(day_slice["close"].iloc[-1])

        options_df = fetch_options_data(symbols[0], str(current_date), str(current_date))
        if "optiontype" in options_df.columns and "option_type" not in options_df.columns:
            options_df = options_df.rename(columns={"optiontype": "option_type"})

        price_dict = {}
        if not options_df.empty:
            for _, row in options_df.iterrows():
                mid = float((row.get("bid", 0.0) + row.get("ask", 0.0)) / 2)
                key = f"{row.get('symbol', symbols[0])}_{row['strike']}_{pd.Timestamp(row['expiration']).date()}_{str(row['option_type']).upper()}"
                price_dict[key] = mid

        portfolio.mark_to_market_options(current_date, price_dict)
        portfolio.settle_expirations(current_date, underlying_prices)

        primary_underlying = underlying_prices.get(symbols[0], 0.0)
        signals = strategy.generate_signals(current_date, options_df, primary_underlying)

        for signal in signals:
            action = str(signal.get("action", "")).upper()
            if action == "BUY":
                pos = OptionPosition(
                    symbol=signal["symbol"],
                    option_type=str(signal["option_type"]).upper(),
                    strike=float(signal["strike"]),
                    expiration=pd.Timestamp(signal["expiration"]).date(),
                    quantity=int(signal["quantity"]),
                    entry_price=float(signal["price"]),
                    entry_date=current_date,
                    entry_delta=signal.get("delta"),
                    entry_gamma=signal.get("gamma"),
                    entry_theta=signal.get("theta"),
                    entry_vega=signal.get("vega"),
                    entry_rho=signal.get("rho"),
                )
                portfolio.open_option(pos)
            elif action == "SELL":
                for pos in portfolio.option_positions:
                    if (
                        pos.symbol == signal["symbol"]
                        and pos.option_type == str(signal["option_type"]).upper()
                        and pos.strike == float(signal["strike"])
                        and pos.expiration == pd.Timestamp(signal["expiration"]).date()
                        and pos.exit_date is None
                    ):
                        portfolio.close_option(pos, float(signal["price"]), current_date)
                        break

        current_date += timedelta(days=1)

    return portfolio
