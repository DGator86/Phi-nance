#!/usr/bin/env python3
"""
Run a walk-forward backtest. With no data in data/bars, uses synthetic 1m bars.

  python -m scripts.run_backtest                    # synthetic, 1 ticker, 1 fold
  python -m scripts.run_backtest --tickers SPY QQQ  # synthetic, 2 tickers
  python -m scripts.run_backtest --data-root data/bars  # use Parquet store if present

Output: per-fold and mean OOS AUC, cone coverage (50/75/90), and gate (mean AUC >= 0.52).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Add project root so phinence is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from phinence.assignment.engine import AssignmentEngine
from phinence.composer.composer import Composer
from phinence.engines.hedge import HedgeEngine
from phinence.engines.liquidity import LiquidityEngine
from phinence.engines.regime import RegimeEngine
from phinence.engines.sentiment import SentimentEngine
from phinence.store.memory_store import InMemoryBarStore
from phinence.store.parquet_store import ParquetBarStore
from phinence.validation.backtest_runner import make_synthetic_bars, run_backtest_fold
from phinence.validation.walk_forward import WFMode, WalkForwardHarness, expanding_windows


def main() -> None:
    p = argparse.ArgumentParser(description="Run WF backtest (synthetic or data/bars).")
    p.add_argument("--tickers", nargs="+", default=["SPY"], help="Tickers to run")
    p.add_argument("--start", default="2023-01-01", help="Start date (synthetic or filter)")
    p.add_argument("--end", default="2024-06-30", help="End date")
    p.add_argument("--data-root", type=str, default="", help="Path to data/bars (Parquet). If empty, use synthetic.")
    p.add_argument("--mode", choices=["intraday", "daily"], default="daily")
    args = p.parse_args()

    if args.data_root and Path(args.data_root).exists():
        bar_store = ParquetBarStore(Path(args.data_root))
        tickers = args.tickers if args.tickers else bar_store.list_tickers()
        if not tickers:
            print("No tickers in store; falling back to synthetic.")
            bar_store = InMemoryBarStore()
            for t in args.tickers:
                df = make_synthetic_bars(t, args.start, args.end, seed=hash(t) % 10000)
                bar_store.put_1m_bars(t, df)
            tickers = args.tickers
    else:
        bar_store = InMemoryBarStore()
        for t in args.tickers:
            df = make_synthetic_bars(t, args.start, args.end, seed=hash(t) % 10000)
            bar_store.put_1m_bars(t, df)
        tickers = args.tickers

    mode = WFMode.DAILY if args.mode == "daily" else WFMode.INTRADAY
    harness = WalkForwardHarness(mode=mode)
    assigner = AssignmentEngine(bar_store)
    composer = Composer()
    engines = {
        "liquidity": LiquidityEngine(),
        "regime": RegimeEngine(),
        "sentiment": SentimentEngine(),
        "hedge": HedgeEngine(),
    }
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    folds = list(expanding_windows(start_ts, end_ts, mode))
    if not folds:
        print("Not enough date range for a single fold.")
        return
    all_metrics: list[dict] = []
    for ticker in tickers:
        for fold in folds:
            m = harness.run_fold(fold, ticker, bar_store, assigner, engines, composer)
            all_metrics.append(m)
            print(f"  {ticker} fold {fold.test_start.date()}–{fold.test_end.date()}: AUC={m['oos_auc']:.3f} cone50/75/90={m['cone_50']:.2%}/{m['cone_75']:.2%}/{m['cone_90']:.2%} n={m.get('n_obs', 0)}")
    mean_auc = sum(x["oos_auc"] for x in all_metrics) / len(all_metrics)
    mean_c75 = sum(x["cone_75"] for x in all_metrics) / len(all_metrics)
    gate = "PASS" if harness.gate_passed(all_metrics) else "FAIL"
    print(f"Mean OOS AUC: {mean_auc:.3f}  Mean 75%% cone: {mean_c75:.2%}  Gate (>=0.52): {gate}")


if __name__ == "__main__":
    main()
