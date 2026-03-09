"""
Direct vectorized backtest — no Lumibot, no datasource.
Uses OHLCV DataFrame directly. Guaranteed to work with pipeline data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phi.exceptions import BacktestError
from phi.logging import get_logger
from phi.backtest.allocation import get_allocation_strategy
from phi.backtest.portfolio import Order, Portfolio
from phi.regime import create_detector_from_params

logger = get_logger(__name__)

_TRADING_MINUTES_PER_YEAR = 252 * 390  # US equity market


def _bars_per_year(df: pd.DataFrame) -> float:
    """Infer annualised bar count from the DataFrame index.

    For daily bars returns 252.  For intraday bars, uses the median bar
    duration so CAGR and Sharpe are correctly annualised regardless of
    the timeframe (1m, 5m, 15m, 1H, etc.).
    """
    if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
        return 252.0
    deltas = pd.Series(df.index.astype("int64")).diff().dropna()
    if deltas.empty:
        return 252.0
    median_ns = float(deltas.median())
    if median_ns <= 0:
        return 252.0
    median_minutes = median_ns / 60e9
    if median_minutes >= 300:       # ≥ 5 hours → treat as daily
        return 252.0
    return _TRADING_MINUTES_PER_YEAR / median_minutes


def run_direct_backtest(
    ohlcv: pd.DataFrame,
    symbol: str,
    indicators: dict[str, dict[str, Any]],
    blend_weights: dict[str, float],
    blend_method: str = "weighted_sum",
    signal_threshold: float = 0.15,
    initial_capital: float = 100_000,
    position_size_pct: float = 0.95,
    regime_series: pd.Series | None = None,
    regime_label_map: dict[str, str] | None = None,
    regime_boosts: dict[str, dict[str, float]] | None = None,
    regime_detector: Any | None = None,
    regime_detector_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """
    Run a vectorized equity backtest directly on OHLCV bars.

    Args:
        ohlcv: Input price bars with open/high/low/close/volume columns.
        symbol: Symbol identifier used in prediction logs.
        indicators: Enabled indicators and their parameter payloads.
        blend_weights: Indicator blend weights.
        blend_method: Blending method name from ``phi.blending.blend_signals``.
        signal_threshold: Absolute threshold required to open/close positions.
        initial_capital: Starting account value.
        position_size_pct: Fraction of available cash used per entry.
        regime_series: Optional per-bar regime labels for regime-weighted blending.
        regime_label_map: Optional mapping from detector labels (for example ``state_0``)
            to canonical regime keys used by ``regime_boosts``.
        regime_boosts: Optional per-regime per-indicator boost multipliers.
        regime_detector: Optional detector instance used to compute ``regime_series``
            when series is not precomputed. Must expose ``predict(ohlcv)``.
        regime_detector_params: Optional detector config payload with shape
            ``{"type": "hmm|kmeans|gmm", "params": {...}}``. When supplied and
            ``regime_detector`` is omitted, the detector is created and fit on ``ohlcv``.

    Returns:
        Tuple of ``(results_dict, strat_like_object)`` where results contain
        ``total_return``, ``cagr``, ``max_drawdown``, ``sharpe``, and ``portfolio_value``.

    results_dict: total_return, cagr, max_drawdown, sharpe, portfolio_value
    strat_like: object with .prediction_log for accuracy display
    """
    if initial_capital <= 0:
        raise BacktestError(f"initial_capital must be > 0, got {initial_capital}")
    position_size_pct = float(np.clip(position_size_pct, 0.01, 1.0))

    df = ohlcv.copy()
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    for r in required:
        if r not in cols:
            raise BacktestError(f"OHLCV missing column: {r}")
    df = df.rename(columns={cols[r]: r for r in required})[required]

    # Compute indicators
    from phi.indicators.simple import INDICATOR_COMPUTERS, compute_indicator

    signals_dict = {}
    for name, cfg in indicators.items():
        if name not in INDICATOR_COMPUTERS:
            continue
        params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
        try:
            sig = compute_indicator(name, df, params)
            if sig is not None and not sig.empty:
                signals_dict[name] = sig
        except Exception as exc:
            logger.warning("run_direct_backtest: indicator %r failed: %s", name, exc)

    if not signals_dict:
        return _empty_results(initial_capital), _empty_strat()

    signals_df = pd.DataFrame(signals_dict)
    signals_df = signals_df.reindex(df.index).ffill().bfill()

    from phi.blending import blend_signals

    if regime_detector is None and regime_detector_params:
        detector_type = str(regime_detector_params.get("type", "")).strip().lower()
        detector_params = dict(regime_detector_params.get("params", {}))
        if detector_type:
            regime_detector = create_detector_from_params(detector_type, detector_params)
            regime_detector.fit(df, window=int(detector_params.get("feature_window", 20)))

    if blend_method == "regime_weighted":
        if regime_series is None and regime_detector is not None:
            regime_series = regime_detector.predict(df)
        if regime_series is None:
            raise BacktestError("regime_series or regime_detector is required when blend_method='regime_weighted'")
        aligned_regimes = regime_series.reindex(df.index).ffill()
        composite = pd.Series(index=signals_df.index, dtype=float, name="composite_signal")
        for idx in signals_df.index:
            regime_value = aligned_regimes.loc[idx] if idx in aligned_regimes.index else np.nan
            if pd.isna(regime_value):
                composite.loc[idx] = 0.0
                continue
            mapped_regime = regime_label_map.get(str(regime_value), str(regime_value)) if regime_label_map else str(regime_value)
            composite.loc[idx] = float(
                blend_signals(
                    signals_df.loc[[idx]],
                    method=blend_method,
                    weights=blend_weights,
                    regime=mapped_regime,
                    regime_boosts=regime_boosts or {},
                ).iloc[0]
            )
    else:
        composite = blend_signals(
            signals_df,
            method=blend_method,
            weights=blend_weights,
            regime=None,
            regime_boosts=None,
        )

    if composite.empty:
        return _empty_results(initial_capital), _empty_strat()

    # Simulate bar-by-bar
    cap = float(initial_capital)
    position = 0  # shares
    portfolio_values: list[float] = [cap]
    prediction_log: list[dict] = []
    closes = df["close"].values

    for i in range(len(composite)):
        sig = composite.iloc[i]
        price = float(closes[i])
        if np.isnan(price) or price <= 0:
            portfolio_values.append(cap)
            continue

        if sig > signal_threshold:
            direction = "UP"
            if position == 0:
                qty = int(cap * position_size_pct // price)
                if qty > 0:
                    position = qty
                    cap -= qty * price
        elif sig < -signal_threshold:
            direction = "DOWN"
            if position > 0:
                cap += position * price
                position = 0
        else:
            direction = "NEUTRAL"

        pv = cap + position * price
        portfolio_values.append(pv)
        prediction_log.append({
            "date": df.index[i],
            "symbol": symbol,
            "signal": direction,
            "price": price,
        })

    # Close any remaining position at last price
    if position > 0:
        cap += position * float(closes[-1])
        position = 0

    pv_series = np.array(portfolio_values)
    returns = np.diff(pv_series) / (pv_series[:-1] + 1e-12)
    total_return = (pv_series[-1] - initial_capital) / initial_capital if initial_capital else 0

    bpy = _bars_per_year(df)
    years = len(df) / bpy if len(df) > 0 else 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(pv_series)
    dd = (peak - pv_series) / (peak + 1e-12)
    max_drawdown = float(np.nanmax(dd)) if len(dd) > 0 else 0.0

    # Sharpe (annualized using actual bar frequency, not hardcoded 252)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bpy))
    else:
        sharpe = 0.0

    results = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "portfolio_value": list(pv_series),
        "net_pl": pv_series[-1] - initial_capital,
    }

    # Strat-like object for _display_results / compute_prediction_accuracy
    strat = type("Strat", (), {"prediction_log": prediction_log, "_prediction_log": prediction_log})()

    return results, strat


def run_portfolio_backtest(
    data_dict: dict[str, pd.DataFrame],
    indicators: dict[str, dict[str, Any]],
    blend_weights: dict[str, float],
    blend_method: str = "weighted_sum",
    initial_capital: float = 100_000,
    allocation_strategy: str = "equal_weight",
    allocation_params: dict[str, Any] | None = None,
    rebalance_frequency: str | int | None = "M",
    rebalance_threshold: float | None = None,
    regime_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Run a simple multi-asset portfolio backtest with periodic/threshold rebalancing."""
    if not data_dict:
        raise BacktestError("data_dict must contain at least one symbol")

    allocation = get_allocation_strategy(allocation_strategy, allocation_params)
    closes: dict[str, pd.Series] = {}
    signal_map: dict[str, pd.Series] = {}
    returns_df = pd.DataFrame()
    for symbol, frame in data_dict.items():
        cols = {c.lower(): c for c in frame.columns}
        if "close" not in cols:
            raise BacktestError(f"{symbol}: missing close column")
        df = frame.rename(columns={cols["close"]: "close"})
        closes[symbol] = df["close"].astype(float)
        if not df.index.is_monotonic_increasing:
            closes[symbol] = closes[symbol].sort_index()

        _, _ = run_direct_backtest(
            ohlcv=frame,
            symbol=symbol,
            indicators=indicators,
            blend_weights=blend_weights,
            blend_method=blend_method,
            signal_threshold=0.0,
            initial_capital=initial_capital,
            position_size_pct=1.0,
            regime_series=regime_series,
        )
        # derive low-cost signal proxy from price return when unavailable
        signal_map[symbol] = closes[symbol].pct_change().fillna(0.0)
        returns_df[symbol] = closes[symbol].pct_change()

    common_index = None
    for s in closes.values():
        common_index = s.index if common_index is None else common_index.union(s.index)
    common_index = common_index.sort_values()
    aligned_prices = pd.DataFrame({sym: s.reindex(common_index).ffill() for sym, s in closes.items()})
    aligned_signals = pd.DataFrame({sym: s.reindex(common_index).ffill().fillna(0.0) for sym, s in signal_map.items()})
    aligned_returns = returns_df.reindex(common_index).ffill().fillna(0.0)

    portfolio = Portfolio(initial_capital=initial_capital)
    transactions: list[dict[str, Any]] = []
    threshold = float(rebalance_threshold) if rebalance_threshold is not None else None

    def should_rebalance(i: int, ts: pd.Timestamp, prev_ts: pd.Timestamp | None, target: dict[str, float]) -> bool:
        if i == 0:
            return True
        if isinstance(rebalance_frequency, int) and rebalance_frequency > 0 and i % rebalance_frequency == 0:
            return True
        if isinstance(rebalance_frequency, str):
            key = rebalance_frequency.upper()
            if key == "D":
                return True
            if prev_ts is not None:
                if key == "W" and ts.isocalendar().week != prev_ts.isocalendar().week:
                    return True
                if key == "M" and (ts.month != prev_ts.month or ts.year != prev_ts.year):
                    return True
                if key == "Q" and (ts.quarter != prev_ts.quarter or ts.year != prev_ts.year):
                    return True
        if threshold is not None and target:
            total = portfolio.total_value()
            if total <= 0:
                return False
            for sym, tw in target.items():
                cw = (portfolio.positions.get(sym, 0.0) * portfolio.current_prices.get(sym, 0.0)) / total
                if abs(cw - tw) > threshold:
                    return True
        return rebalance_frequency in {None, "none", "NONE"} and i == 0

    prev_ts: pd.Timestamp | None = None
    last_target_weights: dict[str, float] = {}
    for i, ts in enumerate(common_index):
        price_row = aligned_prices.loc[ts].dropna()
        prices = {s: float(v) for s, v in price_row.items() if float(v) > 0}
        if not prices:
            continue
        portfolio.update_prices(prices)

        signal_row = aligned_signals.loc[ts].to_dict()
        vol = aligned_returns.loc[:ts].tail(20).std().replace(0, np.nan).to_dict()
        target_weights = allocation.allocate(
            portfolio.total_value(),
            {k: float(v) for k, v in signal_row.items()},
            prices,
            regime=regime_series.loc[ts] if isinstance(regime_series, pd.Series) and ts in regime_series.index else None,
            volatility=vol,
        )

        if should_rebalance(i, pd.Timestamp(ts), prev_ts, target_weights):
            last_target_weights = dict(target_weights)
            total = portfolio.total_value()
            for sym, tw in target_weights.items():
                price = prices.get(sym)
                if not price:
                    continue
                desired_shares = int((total * tw) // price)
                delta = desired_shares - portfolio.positions.get(sym, 0.0)
                if delta:
                    order = Order(symbol=sym, shares=float(delta), price=float(price), timestamp=ts)
                    portfolio.execute_order(order)
                    transactions.append(portfolio.transactions[-1])

        portfolio.record_equity(pd.Timestamp(ts))
        prev_ts = pd.Timestamp(ts)

    if portfolio.positions:
        last_ts = common_index[-1]
        for sym, shares in list(portfolio.positions.items()):
            px = portfolio.current_prices.get(sym)
            if px and shares:
                portfolio.execute_order(Order(symbol=sym, shares=-shares, price=px, timestamp=last_ts))
        portfolio.record_equity(pd.Timestamp(last_ts))

    eq = pd.Series([v for _, v in portfolio.equity_curve], index=[t for t, _ in portfolio.equity_curve], dtype=float)
    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] - initial_capital) / initial_capital if not eq.empty else 0.0
    bpy = _bars_per_year(pd.DataFrame(index=eq.index))
    years = len(eq) / bpy if len(eq) else 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    dd = (eq.cummax() - eq) / eq.cummax().replace(0, np.nan) if not eq.empty else pd.Series(dtype=float)
    sharpe = float(ret.mean() / ret.std() * np.sqrt(bpy)) if len(ret) > 1 and ret.std() > 0 else 0.0

    contributions: dict[str, float] = {}
    for sym in aligned_prices.columns:
        series = aligned_prices[sym].pct_change().fillna(0.0)
        contributions[sym] = float(series.sum())

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(dd.max()) if not dd.empty else 0.0,
        "sharpe": sharpe,
        "portfolio_value": eq.tolist(),
        "transactions": transactions,
        "symbol_contributions": contributions,
        "final_weights": last_target_weights or {s: 0.0 for s in aligned_prices.columns},
    }


def _empty_results(cap: float) -> dict[str, Any]:
    return {
        "total_return": 0,
        "cagr": 0,
        "max_drawdown": 0,
        "sharpe": 0,
        "portfolio_value": [cap],
        "net_pl": 0,
    }


def _empty_strat() -> Any:
    return type("Strat", (), {"prediction_log": []})()
