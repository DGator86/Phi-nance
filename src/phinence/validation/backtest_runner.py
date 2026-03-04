"""
Backtest runner: run projection pipeline over a WF test window and compute OOS metrics.

Uses daily granularity: one projection per test day (as_of = prior close), realized = fwd return.
AUC = directional (up vs down); cone coverage = % of |realized bps| within 50/75/90 cones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pandas as pd

from phinence.contracts.projection_packet import Horizon, ProjectionPacket
from phinence.validation.walk_forward import WFWindow


def _auc_directional(prob_up: list[float], realized_bps: list[float]) -> float:
    """Binary AUC: label = 1 if realized_bps > 0 else 0, score = prob_up. Concordant pairs."""
    if len(prob_up) < 2 or len(realized_bps) != len(prob_up):
        return 0.5
    labels = [1 if r > 0 else 0 for r in realized_bps]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    concordant = 0
    total_pairs = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                total_pairs += 1
                if (labels[i] - labels[j]) * (prob_up[i] - prob_up[j]) > 0:
                    concordant += 1
    return concordant / total_pairs if total_pairs else 0.5


def _cone_coverage(realized_bps: list[float], p50_bps: list[float], p75_bps: list[float], p90_bps: list[float]) -> tuple[float, float, float]:
    """Fraction of |realized_bps| <= cone width. One cone value per observation."""
    if not realized_bps or len(realized_bps) != len(p50_bps):
        return 0.0, 0.0, 0.0
    n = len(realized_bps)
    abs_ret = [abs(r) for r in realized_bps]
    c50 = sum(1 for i in range(n) if abs_ret[i] <= p50_bps[i]) / n
    c75 = sum(1 for i in range(n) if abs_ret[i] <= p75_bps[i]) / n
    c90 = sum(1 for i in range(n) if abs_ret[i] <= p90_bps[i]) / n
    return c50, c75, c90


def run_backtest_fold(
    fold: WFWindow,
    ticker: str,
    bar_store: Any,
    assigner: Any,
    engines: dict[str, str | Any],
    composer: Any,
    horizon: Horizon = Horizon.DAILY,
) -> dict[str, Any]:
    """
    Run pipeline for each day in test window; collect predicted direction and realized return.
    Returns oos_auc, cone_50, cone_75, cone_90.

    engines: dict mapping name -> engine instance, e.g. {"liquidity": LiquidityEngine(), ...}
    """
    from phinence.engines.hedge import HedgeEngine
    from phinence.engines.liquidity import LiquidityEngine
    from phinence.engines.regime import RegimeEngine
    from phinence.engines.sentiment import SentimentEngine
    from phinence.mfm.merger import build_mfm

    liquidity_engine = engines.get("liquidity") or LiquidityEngine()
    regime_engine = engines.get("regime") or RegimeEngine()
    sentiment_engine = engines.get("sentiment") or SentimentEngine()
    hedge_engine = engines.get("hedge") or HedgeEngine()

    df_all = bar_store.read_1m_bars(ticker)
    if df_all.empty or len(df_all) < 100:
        return {
            "fold": fold,
            "ticker": ticker,
            "oos_auc": 0.5,
            "cone_50": 0.0,
            "cone_75": 0.0,
            "cone_90": 0.0,
            "n_obs": 0,
        }
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    test_start = pd.Timestamp(fold.test_start)
    test_end = pd.Timestamp(fold.test_end)
    # Daily bars for realized returns: resample to 1d
    df_1d = df_all.set_index("timestamp").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_1d = df_1d.reset_index()
    test_dates = df_1d[(df_1d["timestamp"] >= test_start) & (df_1d["timestamp"] <= test_end)]["timestamp"].dt.normalize().unique()
    if len(test_dates) < 2:
        return {
            "fold": fold,
            "ticker": ticker,
            "oos_auc": 0.5,
            "cone_50": 0.0,
            "cone_75": 0.0,
            "cone_90": 0.0,
            "n_obs": 0,
        }
    prob_ups: list[float] = []
    realized_bps_list: list[float] = []
    p50_list: list[float] = []
    p75_list: list[float] = []
    p90_list: list[float] = []
    for i in range(len(test_dates) - 1):
        as_of_date = pd.Timestamp(test_dates[i])
        next_date = pd.Timestamp(test_dates[i + 1])
        end_ts = as_of_date + pd.Timedelta(hours=23, minutes=59)
        start_ts = pd.Timestamp(fold.train_start)
        packet = assigner.assign(ticker, as_of_date.to_pydatetime(), start_ts=start_ts, end_ts=end_ts)
        liq = liquidity_engine.run(packet)
        reg = regime_engine.run(packet)
        sent = sentiment_engine.run(packet)
        hed = hedge_engine.run(packet)
        mfm = build_mfm(ticker, as_of_date.to_pydatetime(), liquidity=liq, regime=reg, sentiment=sent, hedge=hed)
        proj: ProjectionPacket = composer.run(mfm, horizons=[horizon])
        hp = proj.get_horizon(horizon)
        if not hp:
            continue
        prob_ups.append(hp.direction.up)
        p50_list.append(hp.cone.p50_bps)
        p75_list.append(hp.cone.p75_bps)
        p90_list.append(hp.cone.p90_bps)
        close_today = df_1d[df_1d["timestamp"].dt.normalize() == as_of_date]["close"].iloc[-1]
        close_next = df_1d[df_1d["timestamp"].dt.normalize() == next_date]["close"].iloc[-1]
        if close_today and close_today != 0:
            realized_bps_list.append((close_next - close_today) / close_today * 10_000)
        else:
            realized_bps_list.append(0.0)
    if not prob_ups:
        return {
            "fold": fold,
            "ticker": ticker,
            "oos_auc": 0.5,
            "cone_50": 0.0,
            "cone_75": 0.0,
            "cone_90": 0.0,
            "n_obs": 0,
        }
    oos_auc = _auc_directional(prob_ups, realized_bps_list)
    c50, c75, c90 = _cone_coverage(realized_bps_list, p50_list, p75_list, p90_list)
    return {
        "fold": fold,
        "ticker": ticker,
        "oos_auc": oos_auc,
        "cone_50": c50,
        "cone_75": c75,
        "cone_90": c90,
        "n_obs": len(prob_ups),
    }


def make_synthetic_bars(
    ticker: str,
    start_date: str,
    end_date: str,
    bars_per_day: int = 390,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate 1m bars (random walk) for backtest without real data."""
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    dr = pd.date_range(start=start_date, end=end_date, freq="B")
    rows = []
    price = 100.0
    for d in dr:
        for i in range(bars_per_day):
            ts = d + pd.Timedelta(minutes=i)
            ret = np.random.randn() * 0.001
            open_p = price
            price = price * (1 + ret)
            high = max(open_p, price) * (1 + np.abs(np.random.randn()) * 0.0005)
            low = min(open_p, price) * (1 - np.abs(np.random.randn()) * 0.0005)
            vol = int(np.random.lognormal(0, 2) * 1000)
            rows.append({
                "timestamp": ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": price,
                "volume": vol,
            })
    return pd.DataFrame(rows)
