#!/usr/bin/env python3
"""
scripts/fetch_data.py
======================
CLI entry point for fetching and caching OHLCV market data.

Usage
-----
    python scripts/fetch_data.py \\
        --symbol SPY QQQ BTC-USD \\
        --start  2020-01-01 \\
        --end    2024-12-31 \\
        --tf     1D \\
        --vendor yfinance

    # List cached datasets
    python scripts/fetch_data.py --list

    # Clear cache for a symbol
    python scripts/fetch_data.py --clear --symbol SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phi-nance data fetcher / cache manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol",  nargs="+", default=["SPY"], help="One or more tickers")
    p.add_argument("--start",   default="2022-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",     default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--tf",      default="1D", help="Timeframe")
    p.add_argument("--vendor",  default="yfinance",
                   choices=["yfinance", "alphavantage", "binance"])
    p.add_argument("--api-key", default="", help="Alpha Vantage / Binance API key")
    p.add_argument("--list",    action="store_true", help="List all cached datasets and exit")
    p.add_argument("--force",   action="store_true", help="Force re-fetch even if cached")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from phinance.data import fetch_and_cache, list_cached_datasets

    if args.list:
        datasets = list_cached_datasets()
        if not datasets:
            print("No cached datasets found.")
            return 0
        print(f"\n{'Vendor':<14} {'Symbol':<12} {'TF':<6} {'Start':<12} {'End':<12} {'Rows':>6}")
        print("─" * 66)
        for d in datasets:
            print(
                f"{d['vendor']:<14} {d['symbol']:<12} {d['timeframe']:<6} "
                f"{d['start']:<12} {d['end']:<12} {d['rows']:>6,}"
            )
        print()
        return 0

    print(f"\n  Fetching {len(args.symbol)} symbol(s)  [{args.tf}  {args.start}→{args.end}]\n")
    ok = 0
    for sym in args.symbol:
        try:
            df = fetch_and_cache(
                vendor    = args.vendor,
                symbol    = sym,
                timeframe = args.tf,
                start     = args.start,
                end       = args.end,
                api_key   = args.api_key or None,
            )
            print(f"  ✅  {sym:<10}  {len(df):>6,} bars   "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
            ok += 1
        except Exception as exc:
            print(f"  ❌  {sym:<10}  FAILED: {exc}")

    print(f"\n  {ok}/{len(args.symbol)} symbols fetched successfully.\n")
    return 0 if ok == len(args.symbol) else 1


if __name__ == "__main__":
    sys.exit(main())
