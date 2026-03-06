"""
phinance.backtest.runner
=========================

High-level backtest orchestrator.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from phinance.backtest.engine import simulate
from phinance.backtest.metrics import compute_all
from phinance.backtest.models import BacktestResult
from phinance.blending.blender import blend_signals
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
from phinance.exceptions import InsufficientDataError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_ROWS = 10
_MAX_INDICATOR_CACHE = 128
_INDICATOR_CACHE: "OrderedDict[Tuple[str, Tuple[Tuple[str, Any], ...], int, Any, Any], pd.Series]" = OrderedDict()


def _cache_key(name: str, params: Dict[str, Any], df: pd.DataFrame) -> Tuple[str, Tuple[Tuple[str, Any], ...], int, Any, Any]:
    sorted_params = tuple(sorted((str(k), str(v)) for k, v in params.items()))
    return name, sorted_params, len(df), df.index[0], df.index[-1]


def _compute_indicator_cached(name: str, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    key = _cache_key(name, params, df)
    cached = _INDICATOR_CACHE.get(key)
    if cached is not None:
        _INDICATOR_CACHE.move_to_end(key)
        return cached

    result = compute_indicator(name, df, params)
    _INDICATOR_CACHE[key] = result
    _INDICATOR_CACHE.move_to_end(key)

    while len(_INDICATOR_CACHE) > _MAX_INDICATOR_CACHE:
        _INDICATOR_CACHE.popitem(last=False)

    return result


def run_backtest(
    ohlcv: pd.DataFrame,
    symbol: str = "UNKNOWN",
    indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    blend_weights: Optional[Dict[str, float]] = None,
    blend_method: str = "weighted_sum",
    signal_threshold: float = 0.15,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.95,
    regime_probs: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    if ohlcv is None or len(ohlcv) < _MIN_ROWS:
        raise InsufficientDataError(
            f"Backtest requires at least {_MIN_ROWS} rows; got "
            f"{len(ohlcv) if ohlcv is not None else 0}."
        )

    df = _normalise_df(ohlcv)

    signals_dict: Dict[str, pd.Series] = {}
    active = indicators or {}
    for name, cfg in active.items():
        if not cfg.get("enabled", True):
            continue
        if name not in INDICATOR_CATALOG:
            continue
        params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
        try:
            sig = _compute_indicator_cached(name, df, params)
            if sig is not None and not sig.empty:
                signals_dict[name] = sig
        except Exception as exc:
            logger.warning("Indicator '%s' failed: %s", name, exc)

    if not signals_dict:
        logger.warning("No valid indicators; returning empty result.")
        return _empty_result(symbol, initial_capital)

    signals_df = pd.DataFrame(signals_dict, index=df.index).ffill().bfill()

    composite = blend_signals(
        signals=signals_df,
        weights=blend_weights or {},
        method=blend_method,
        regime_probs=regime_probs,
    )
    if composite.empty:
        return _empty_result(symbol, initial_capital)

    portfolio_values, prediction_log, trades = simulate(
        ohlcv=df,
        composite_signal=composite,
        symbol=symbol,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        signal_threshold=signal_threshold,
    )

    metrics = compute_all(portfolio_values, df, initial_capital, trades)

    return BacktestResult(
        symbol=symbol,
        total_return=metrics["total_return"],
        cagr=metrics["cagr"],
        max_drawdown=metrics["max_drawdown"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        win_rate=metrics["win_rate"],
        total_trades=len(trades),
        portfolio_value=portfolio_values,
        net_pl=metrics["net_pl"],
        trades=trades,
        prediction_log=prediction_log,
        metadata={
            "symbol": symbol,
            "initial_capital": initial_capital,
            "blend_method": blend_method,
            "indicators": list(signals_dict.keys()),
        },
    )


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise InsufficientDataError(f"OHLCV DataFrame missing columns: {missing}")
    return df.rename(columns={cols[c]: c for c in required})[required]


def _empty_result(symbol: str, initial_capital: float) -> BacktestResult:
    return BacktestResult(
        symbol=symbol,
        portfolio_value=[initial_capital],
        metadata={"initial_capital": initial_capital},
    )
