#!/usr/bin/env python3
"""
Phase 2 data spine: backfill historical 1m bars, persist live Tradier bars, pull FINRA short volume.

Requires .env with:
  POLYGON_API_KEY=...   (or MASSIVE_API_KEY=...) for historical 1m bars
  TRADIER_ACCESS_TOKEN=... for live bars (optional for backfill-only)

Run from repo root:
  python -m scripts.run_phase2
  python -m scripts.run_phase2 --tickers SPY QQQ AAPL --years 2
  python -m scripts.run_phase2 --short-volume-only
  python -m scripts.run_phase2 --skip-backfill   # only live + short volume

If an API returns 401/403: key is invalid or expired. Rotate keys and set in .env (never commit .env).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env before any provider reads os.environ
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def _check_polygon_key() -> str | None:
    return os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")


def _check_tradier_key() -> str | None:
    return os.environ.get("TRADIER_ACCESS_TOKEN")


def main() -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Phase 2: backfill bars, live bars, FINRA short volume")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"], help="Tickers for backfill")
    p.add_argument("--years", type=int, default=2, help="Years of history to backfill")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bars store root")
    p.add_argument("--short-volume-only", action="store_true", help="Only fetch FINRA short volume")
    p.add_argument("--skip-backfill", action="store_true", help="Skip Polygon backfill")
    p.add_argument("--use-massive", action="store_true", help="Use massive client if installed")
    p.add_argument("--check-gaps", action="store_true", help="After backfill, run no-gap >5 bars sanity check on bar store")
    args = p.parse_args()

    # ---------- FINRA short volume (no key) ----------
    print("Phase 2C: FINRA short volume...")
    try:
        from data.providers import finra_short_volume
        path = finra_short_volume.fetch_and_persist_daily()
        path = finra_short_volume.fetch_and_persist_daily()
        if path:
            print(f"  Saved: {path}")
        else:
            print("  No data (file not yet posted or date is weekend?).")
    except Exception as e:
        print(f"  FINRA error: {e}")
    if args.short_volume_only:
        return 0

    # ---------- Historical 1m backfill ----------
    if not args.skip_backfill:
        polygon_key = _check_polygon_key()
        if not polygon_key:
            print("\n*** Backfill skipped: no Polygon/Massive API key. ***")
            print("  Set in .env:")
            print("    POLYGON_API_KEY=your_key")
            print("  or  MASSIVE_API_KEY=your_key")
            print("  Then run again. Get a key at polygon.io or massive.com")
        else:
            print("\nPhase 2A: Historical 1m bars (Polygon/Massive)...")
            from datetime import date
            from data.providers import polygon_backfill
            store_root = args.data_root
            today = date.today()
            for ticker in args.tickers:
                for y in range(today.year - args.years, today.year + 1):
                    try:
                        out = polygon_backfill.backfill_ticker_year(
                            ticker, y,
                            store_root=store_root,
                            api_key=polygon_key,
                            use_massive_client=args.use_massive,
                        )
                        if out:
                            print(f"  {ticker} {y}: {out}")
                        else:
                            print(f"  {ticker} {y}: no data")
                    except Exception as e:
                        err = str(e).lower()
                        if "401" in err or "403" in err or "unauthorized" in err or "forbidden" in err:
                            print(f"  *** {ticker} {y}: API key rejected (401/403). Check key and .env. ***")
                        else:
                            print(f"  {ticker} {y}: {e}")

        if args.check_gaps and args.tickers:
            print("\nPhase 2A sanity check: no-gap >5 bars...")
            try:
                from phinence.store.parquet_store import ParquetBarStore, check_no_gap_more_than_n_bars
                store = ParquetBarStore(args.data_root)
                for ticker in args.tickers:
                    df = store.read_1m_bars(ticker)
                    if df.empty:
                        print(f"  {ticker}: no data")
                        continue
                    ok = check_no_gap_more_than_n_bars(df, max_gap=5)
                    print(f"  {ticker}: no-gap check {'PASS' if ok else 'FAIL'} (rows={len(df)})")
            except Exception as e:
                print(f"  Check error: {e}")

    # ---------- Live Tradier bars (persist today into same schema) ----------
    tradier_key = _check_tradier_key()
    if not tradier_key:
        print("\nPhase 2B: Live bars skipped (no TRADIER_ACCESS_TOKEN in .env).")
    else:
        print("\nPhase 2B: Live 1m bars (Tradier)...")
        try:
            from data.providers import tradier
            import pandas as pd
            import pyarrow.parquet as pq
            from phinence.store.schemas import BAR_1M_SCHEMA
            import pyarrow as pa
            store_root = args.data_root
            for ticker in args.tickers:
                df = tradier.fetch_live_1m_bars(ticker, api_key=tradier_key)
                if df.empty:
                    print(f"  {ticker}: no live bars")
                    continue
                # Persist to data/bars/{ticker}/live_{today}.parquet or append to year
                today_str = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
                year = pd.Timestamp.now().year
                ticker_dir = store_root / ticker.upper()
                ticker_dir.mkdir(parents=True, exist_ok=True)
                path = ticker_dir / f"live_{today_str.replace('-', '')}.parquet"
                table = pa.Table.from_pandas(df, schema=BAR_1M_SCHEMA, preserve_index=False)
                pq.write_table(table, path)
                print(f"  {ticker}: {len(df)} bars -> {path}")
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "403" in err or "unauthorized" in err:
                print(f"  *** Tradier key rejected. Check TRADIER_ACCESS_TOKEN in .env. ***")
            else:
                print(f"  Tradier error: {e}")

    print("\nPhase 2 run complete. If backfill failed: set POLYGON_API_KEY (or MASSIVE_API_KEY) in .env and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
