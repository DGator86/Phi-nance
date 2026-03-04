#!/usr/bin/env python3
"""
Phase 6.5 sandbox: run daily projection pipeline and save ProjectionPackets.

  python -m scripts.run_paper_daily
  python -m scripts.run_paper_daily --tickers SPY QQQ --data-root data/bars --date 2024-01-15

Uses bar_store (Parquet or synthetic if no data). Writes data/paper_packets/YYYY-MM-DD/{ticker}.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, date, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def main() -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Phase 6.5: daily paper packets")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"], help="Tickers")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bars store")
    p.add_argument("--packets-dir", type=Path, default=REPO_ROOT / "data" / "paper_packets", help="Output dir for packets")
    p.add_argument("--date", type=str, default="", help="As-of date YYYY-MM-DD; default today")
    args = p.parse_args()

    as_of = datetime.now(timezone.utc)
    if args.date:
        try:
            d = datetime.strptime(args.date, "%Y-%m-%d")
            as_of = d.replace(hour=16, minute=0, second=0, microsecond=0)
        except ValueError:
            print(f"Invalid --date {args.date}; use YYYY-MM-DD")
            return 1

    if args.data_root.exists():
        from phinence.store.parquet_store import ParquetBarStore
        bar_store = ParquetBarStore(args.data_root)
        tickers = [t for t in args.tickers if t in bar_store.list_tickers()] or args.tickers
    else:
        from phinence.store.memory_store import InMemoryBarStore
        from phinence.validation.backtest_runner import make_synthetic_bars
        import pandas as pd
        bar_store = InMemoryBarStore()
        end = pd.Timestamp(as_of)
        start = end - pd.Timedelta(days=365)
        for t in args.tickers:
            bar_store.put_1m_bars(t, make_synthetic_bars(t, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), seed=hash(t) % 10000))
        tickers = args.tickers

    from phinence.assignment.engine import AssignmentEngine
    from phinence.composer.composer import Composer
    from phinence.engines.hedge import HedgeEngine
    from phinence.engines.liquidity import LiquidityEngine
    from phinence.engines.regime import RegimeEngine
    from phinence.engines.sentiment import SentimentEngine
    from phinence.validation.paper_trading import paper_run_daily, save_packets

    assigner = AssignmentEngine(bar_store)
    composer = Composer()
    engines = {
        "liquidity": LiquidityEngine(),
        "regime": RegimeEngine(),
        "sentiment": SentimentEngine(),
        "hedge": HedgeEngine(),
    }

    packets = paper_run_daily(tickers, bar_store, assigner, engines, composer, as_of=as_of)
    date_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)[:10]
    out_dir = args.packets_dir / date_str
    paths = save_packets(packets, out_dir)
    print(f"Saved {len(paths)} packets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
