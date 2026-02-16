#!/usr/bin/env python3
"""
Phase 6.5 dashboard: load saved paper packets, compute 20d AUC, 75% cone, 20d IC, kill status.

  python -m scripts.run_dashboard
  python -m scripts.run_dashboard --packets-dir data/paper_packets --data-root data/bars --days 30

Requires --data-root with bar data to compute realized returns (AUC/IC). Without it, only packet counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _daily_close_series(bar_store, ticker: str) -> pd.Series | None:
    """Resample 1m bars to 1d close; return series index=date, value=close."""
    df = bar_store.read_1m_bars(ticker)
    if df is None or df.empty or "timestamp" not in df.columns or "close" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    daily = df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
    daily.index = daily.index.normalize()
    return daily["close"]


def _realized_bps(bar_store, ticker: str, as_of_date: pd.Timestamp) -> float | None:
    """Next trading day return in bps; None if missing."""
    series = _daily_close_series(bar_store, ticker)
    if series is None or series.empty:
        return None
    try:
        as_of_norm = as_of_date.normalize()
        if as_of_norm not in series.index:
            return None
        # Next trading day
        later = series.index[series.index > as_of_norm]
        if len(later) == 0:
            return None
        next_d = later[0]
        c_today = series.loc[as_of_norm]
        c_next = series.loc[next_d]
        if c_today == 0:
            return None
        return (float(c_next) - float(c_today)) / float(c_today) * 10_000
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 6.5 dashboard: 20d AUC, cone, IC, kill status")
    p.add_argument("--packets-dir", type=Path, default=REPO_ROOT / "data" / "paper_packets", help="Paper packets root")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bars store for realized returns")
    p.add_argument("--days", type=int, default=30, help="Max days of packet history to load")
    p.add_argument("--window", type=int, default=20, help="Rolling window for 20d AUC / cone / IC")
    args = p.parse_args()

    if not args.packets_dir.exists():
        print("No packets dir:", args.packets_dir)
        return 0

    from phinence.contracts.projection_packet import Horizon
    from phinence.validation.backtest_runner import _auc_directional, _cone_coverage
    from phinence.validation.paper_trading import (
        PaperMetrics,
        load_packets_from_dir,
        should_kill_paper,
    )

    date_dirs = sorted([d for d in args.packets_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)[: args.days]
    if not date_dirs:
        print("No date subdirs in", args.packets_dir)
        return 0

    bar_store = None
    if args.data_root.exists():
        try:
            from phinence.store.parquet_store import ParquetBarStore
            bar_store = ParquetBarStore(args.data_root)
        except Exception:
            pass

    # By date: list of (packet, realized_bps)
    by_date: list[tuple[str, list[tuple[Any, float]]]] = []
    for d in reversed(date_dirs):
        packets = load_packets_from_dir(d)
        if not packets:
            continue
        date_str = d.name
        as_of_ts = pd.Timestamp(date_str)
        rows: list[tuple[Any, float]] = []
        for p in packets:
            hp = p.get_horizon(Horizon.DAILY)
            if not hp:
                continue
            prob_up = hp.direction.up
            p50, p75, p90 = hp.cone.p50_bps, hp.cone.p75_bps, hp.cone.p90_bps
            realized = None
            if bar_store:
                realized = _realized_bps(bar_store, p.ticker, as_of_ts)
            if realized is None:
                continue
            rows.append((p, realized, prob_up, p50, p75, p90))
        if not rows:
            continue
        by_date.append((date_str, rows))

    if not by_date:
        # No packet dirs or no packets with realized returns
        any_packets = False
        for d in date_dirs:
            packets = load_packets_from_dir(d)
            if packets:
                any_packets = True
                print(f"  {d.name}: {len(packets)} packets (no bar store for realized returns)")
        if any_packets:
            print("Set --data-root to bar store to compute 20d AUC, cone coverage, and IC.")
        else:
            print("No packet date dirs with JSON files in", args.packets_dir)
        return 0

    # Daily metrics
    metrics_history: list[PaperMetrics] = []
    for date_str, rows in by_date:
        prob_ups = [r[2] for r in rows]
        realized = [r[1] for r in rows]
        p50s, p75s, p90s = [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
        auc = _auc_directional(prob_ups, realized)
        c50, c75, c90 = _cone_coverage(realized, p50s, p75s, p90s)
        n = len(realized)
        if n >= 2:
            ic = float(pd.Series(prob_ups).corr(pd.Series(realized))) if n else 0.0
        else:
            ic = 0.0
        metrics_history.append(
            PaperMetrics(window_days=args.window, auc=auc, cone_75_coverage=c75, ic=ic)
        )

    # Rolling 20d over last `window` days
    window = min(args.window, len(by_date))
    if window > 0:
        flat = []
        for date_str, rows in by_date[-window:]:
            for r in rows:
                flat.append((r[2], r[1], r[4]))  # prob_up, realized_bps, p75
        if len(flat) >= 2:
            prob_ups = [x[0] for x in flat]
            realized = [x[1] for x in flat]
            p75s = [x[2] for x in flat]
            p50s = [x[2] * 0.5 for x in flat]
            p90s = [x[2] * 1.2 for x in flat]
            roll_auc = _auc_directional(prob_ups, realized)
            _, roll_c75, _ = _cone_coverage(realized, p50s, p75s, p90s)
            n = len(realized)
            roll_ic = float(pd.Series(prob_ups).corr(pd.Series(realized))) if n >= 2 else 0.0
        else:
            roll_auc, roll_c75, roll_ic = 0.5, 0.0, 0.0
    else:
        roll_auc, roll_c75, roll_ic = 0.5, 0.0, 0.0

    print("--- Phase 6.5 Dashboard ---")
    print(f"Dates loaded: {len(by_date)}  (last: {by_date[-1][0]})")
    print(f"{args.window}d rolling  AUC: {roll_auc:.3f}  75%% cone: {roll_c75:.2%}  20d IC: {roll_ic:.3f}")
    kill, reasons = should_kill_paper(metrics_history)
    print(f"Kill gate: {'FAIL - ' + '; '.join(reasons) if kill else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
