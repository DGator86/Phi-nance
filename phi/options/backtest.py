"""Options backtesting utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from phi.run_config import RunConfig
from phi.logging import get_logger

from .contract import OptionContract, OptionType
from .position import OptionPosition
from .pricing import black_scholes_price, delta, gamma, theta, vega

logger = get_logger(__name__)


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
) -> Dict[str, float]:
    """Return Black-Scholes greeks as a plain dictionary."""
    opt_type = _parse_option_type(option_type)
    g = OptionGreeks(
        delta=delta(S, K, T, r, sigma, opt_type),
        gamma=gamma(S, K, T, r, sigma),
        theta=theta(S, K, T, r, sigma, opt_type),
        vega=vega(S, K, T, r, sigma),
    )
    return asdict(g)


def _parse_option_type(value: OptionType | str) -> OptionType:
    if isinstance(value, OptionType):
        return value
    return OptionType(str(value).lower())


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _compute_metrics(values: Iterable[float], initial_capital: float) -> Dict[str, float]:
    pv = np.array(list(values), dtype=float)
    if pv.size == 0:
        pv = np.array([initial_capital], dtype=float)

    total_return = float((pv[-1] - initial_capital) / initial_capital) if initial_capital else 0.0
    n_years = max((len(pv) - 1) / 252.0, 1 / 252.0)
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)

    peaks = np.maximum.accumulate(pv)
    drawdowns = (pv - peaks) / np.where(peaks == 0, 1.0, peaks)
    max_dd = float(np.min(drawdowns))

    rets = np.diff(pv) / np.where(pv[:-1] == 0, 1.0, pv[:-1])
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252.0)) if rets.size > 1 and np.std(rets) > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def run_options_backtest(
    data_or_config: pd.DataFrame | RunConfig,
    data: Optional[pd.DataFrame] = None,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """Run a single-position European options backtest.

    Preferred call:
        run_options_backtest(config: RunConfig, data: pd.DataFrame)

    Backward compatible legacy call:
        run_options_backtest(ohlcv, symbol="SPY", strategy_type="long_call", ...)
    """
    if isinstance(data_or_config, RunConfig):
        if data is None:
            raise ValueError("data is required when passing a RunConfig")
        return _run_from_config(data_or_config, data)

    return _run_legacy(data_or_config, **legacy_kwargs)


def _run_from_config(config: RunConfig, data: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Starting options backtest for symbols=%s from %s to %s", config.symbols, config.start_date, config.end_date)
    if data is None or data.empty:
        raise ValueError("data must contain underlying OHLCV history")
    if len(config.symbols) != 1:
        raise ValueError("options backtest currently supports exactly one symbol")

    symbol = config.symbols[0]
    params = config.option_params.get(symbol)
    if not params:
        raise ValueError(f"option_params missing for symbol '{symbol}'")

    option_type = _parse_option_type(params["option_type"])
    contract = OptionContract(
        underlying=symbol,
        option_type=option_type,
        strike=float(params["strike"]),
        expiry=_to_date(params["expiry"]),
        style=str(params.get("style", "european")),
        multiplier=int(params.get("multiplier", 100)),
    )
    iv = float(params.get("iv", 0.3))
    if iv <= 0 or iv > 5:
        logger.warning("IV %.4f for %s is outside expected range (0, 5]", iv, symbol)
    r = float(params.get("r", 0.02))
    quantity = int(params.get("quantity", 1))

    close = data["close"].astype(float)
    cash = float(config.initial_capital)
    position: OptionPosition | None = None
    trade_log: list[dict[str, Any]] = []
    portfolio_values: list[float] = []

    for ts, price in close.items():
        as_of = _to_date(ts)

        if position is None and as_of < contract.expiry:
            premium = black_scholes_price(price, contract.strike, contract.time_to_expiry(as_of), r, iv, contract.option_type)
            total_cost = premium * quantity * contract.multiplier
            if cash < total_cost:
                logger.error("Insufficient capital: cash=%s required=%s", cash, total_cost)
                raise ValueError("insufficient capital to open options position")
            cash -= total_cost
            position = OptionPosition(contract=contract, quantity=quantity, entry_cost=total_cost)
            logger.info("Opened %s position: symbol=%s strike=%.2f expiry=%s qty=%d premium=%.4f", contract.option_type.value, symbol, contract.strike, contract.expiry, quantity, premium)
            trade_log.append({"date": as_of.isoformat(), "action": "buy", "price": premium, "value": total_cost})

        position_value = position.mark_to_market(price, as_of, r, iv) if position else 0.0
        portfolio_values.append(cash + position_value)

        if position and as_of >= contract.expiry:
            intrinsic = max(price - contract.strike, 0.0) if contract.option_type == OptionType.CALL else max(contract.strike - price, 0.0)
            settlement = intrinsic * position.quantity * contract.multiplier
            cash += settlement
            logger.info("Settled option at expiry: symbol=%s intrinsic=%.4f settlement=%.4f", symbol, intrinsic, settlement)
            trade_log.append({"date": as_of.isoformat(), "action": "expiry", "price": intrinsic, "value": settlement})
            position = None

    final_value = cash + (position.mark_to_market(float(close.iloc[-1]), _to_date(close.index[-1]), r, iv) if position else 0.0)
    metrics = _compute_metrics(portfolio_values, config.initial_capital)
    logger.info("Completed options backtest for %s: final_value=%.2f trades=%d", symbol, final_value, len(trade_log))
    return {
        "portfolio_value": portfolio_values,
        "final_value": final_value,
        "trades": trade_log,
        **metrics,
    }


def _run_legacy(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    strategy_type: str = "long_call",
    initial_capital: float = 100_000.0,
    **_: Any,
) -> Dict[str, Any]:
    if ohlcv is None or ohlcv.empty:
        return {"portfolio_value": [initial_capital], "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": []}

    option_type = OptionType.CALL if strategy_type == "long_call" else OptionType.PUT
    expiry = _to_date(ohlcv.index[-1])
    strike = float(ohlcv["close"].iloc[0])

    cfg = RunConfig(
        symbols=[symbol],
        start_date=_to_date(ohlcv.index[0]),
        end_date=_to_date(ohlcv.index[-1]),
        initial_capital=initial_capital,
        trading_mode="options",
        option_params={
            symbol: {
                "option_type": option_type.value,
                "strike": strike,
                "expiry": expiry,
                "iv": 0.3,
                "r": 0.02,
                "multiplier": 100,
                "quantity": 1,
            }
        },
    )
    return _run_from_config(cfg, ohlcv)
