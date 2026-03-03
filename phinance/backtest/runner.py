"""
phinance.backtest.runner
=========================

High-level backtest orchestrator.

``run_backtest()`` is the single entry point for all equities backtests.
It accepts OHLCV + indicator config, orchestrates signal computation,
blending, and simulation, and returns a rich ``BacktestResult``.

Usage
-----
    from phinance.backtest import run_backtest

    result = run_backtest(
        ohlcv      = df,
        symbol     = "SPY",
        indicators = {"RSI": {"enabled": True, "params": {"period": 14}}},
        blend_weights  = {"RSI": 1.0},
        blend_method   = "weighted_sum",
        initial_capital = 100_000,
    )
    print(f"CAGR: {result.cagr:.1%}  Sharpe: {result.sharpe:.2f}")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from phinance.backtest.engine import simulate
from phinance.backtest.metrics import compute_all, bars_per_year
from phinance.backtest.models import BacktestResult
from phinance.blending.blender import blend_signals
from phinance.strategies.indicator_catalog import compute_indicator, INDICATOR_CATALOG
from phinance.exceptions import InsufficientDataError
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_ROWS = 10


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
    """Run a vectorised equities backtest and return a BacktestResult.

    Parameters
    ----------
    ohlcv             : pd.DataFrame — OHLCV with DatetimeIndex
    symbol            : str          — ticker label for trades
    indicators        : dict         — ``{name: {"enabled": bool, "params": dict}}``
    blend_weights     : dict         — ``{name: weight}``
    blend_method      : str          — one of BLEND_METHODS
    signal_threshold  : float        — minimum signal to act (default 0.15)
    initial_capital   : float        — starting NAV (default 100 000)
    position_size_pct : float        — fraction of capital per trade (default 0.95)
    regime_probs      : pd.DataFrame — optional regime probabilities for
                                       regime_weighted blending

    Returns
    -------
    BacktestResult
    """
    if ohlcv is None or len(ohlcv) < _MIN_ROWS:
        raise InsufficientDataError(
            f"Backtest requires at least {_MIN_ROWS} rows; got "
            f"{len(ohlcv) if ohlcv is not None else 0}."
        )

    # Normalise OHLCV columns
    df = _normalise_df(ohlcv)

    # Compute indicator signals
    signals_dict: Dict[str, pd.Series] = {}
    active = indicators or {}
    for name, cfg in active.items():
        if not cfg.get("enabled", True):
            continue
        if name not in INDICATOR_CATALOG:
            continue
        params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
        try:
            sig = compute_indicator(name, df, params)
            if sig is not None and not sig.empty:
                signals_dict[name] = sig
        except Exception as exc:
            logger.warning("Indicator '%s' failed: %s", name, exc)

    if not signals_dict:
        logger.warning("No valid indicators; returning empty result.")
        return _empty_result(symbol, initial_capital)

    signals_df = pd.DataFrame(signals_dict).reindex(df.index).ffill().bfill()

    # Blend
    composite = blend_signals(
        signals=signals_df,
        weights=blend_weights or {},
        method=blend_method,
        regime_probs=regime_probs,
    )
    if composite.empty:
        return _empty_result(symbol, initial_capital)

    # Simulate
    portfolio_values, prediction_log, trades = simulate(
        ohlcv=df,
        composite_signal=composite,
        symbol=symbol,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        signal_threshold=signal_threshold,
    )

    # Compute metrics
    metrics = compute_all(portfolio_values, df, initial_capital, trades)

    return BacktestResult(
        symbol          = symbol,
        total_return    = metrics["total_return"],
        cagr            = metrics["cagr"],
        max_drawdown    = metrics["max_drawdown"],
        sharpe          = metrics["sharpe"],
        sortino         = metrics["sortino"],
        win_rate        = metrics["win_rate"],
        total_trades    = len(trades),
        portfolio_value = portfolio_values,
        net_pl          = metrics["net_pl"],
        trades          = trades,
        prediction_log  = prediction_log,
        metadata        = {
            "symbol":          symbol,
            "initial_capital": initial_capital,
            "blend_method":    blend_method,
            "indicators":      list(signals_dict.keys()),
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are lowercase."""
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise InsufficientDataError(
            f"OHLCV DataFrame missing columns: {missing}"
        )
    return df.rename(columns={cols[c]: c for c in required})[required]


def _empty_result(symbol: str, initial_capital: float) -> BacktestResult:
    return BacktestResult(
        symbol          = symbol,
        portfolio_value = [initial_capital],
        metadata        = {"initial_capital": initial_capital},
    )
