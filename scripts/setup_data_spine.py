#!/usr/bin/env python3
"""Bootstrap Phase 2 data spine with optional fallback sample data."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from phinence.store.parquet_store import ParquetBarStore, check_no_gap_more_than_n_bars

LOGGER = logging.getLogger("setup_data_spine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate and validate Phase 2 data spine")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"], help="Tickers to backfill/verify")
    parser.add_argument("--years", type=int, default=2, help="Years for scripts.run_phase2 backfill")
    parser.add_argument("--use-massive", action="store_true", help="Pass --use-massive to scripts.run_phase2")
    parser.add_argument("--max-gap", type=int, default=5, help="Maximum allowed gap (in bars)")
    parser.add_argument("--phase2-retries", type=int, default=2, help="Retry attempts for scripts.run_phase2")
    parser.add_argument("--sample-only", action="store_true", help="Skip API runs and write sample data only")
    parser.add_argument("--skip-sample", action="store_true", help="Do not write sample data fallback")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)
        LOGGER.info("Loaded environment from %s", env_path)
    elif not env_path.exists():
        LOGGER.warning("Missing .env file. Copy .env.example to .env for API-based population.")


def _has_parquet_files(path: Path) -> bool:
    return any(path.rglob("*.parquet"))


def _run_phase2(args: argparse.Namespace) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_phase2",
        "--tickers",
        *[t.upper() for t in args.tickers],
        "--years",
        str(args.years),
        "--check-gaps",
    ]
    if args.use_massive:
        cmd.append("--use-massive")

    attempts = max(args.phase2_retries, 1)
    for attempt in range(1, attempts + 1):
        LOGGER.info("Running Phase 2 (%d/%d): %s", attempt, attempts, " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode == 0:
            return True
        LOGGER.warning("scripts.run_phase2 failed with exit code %s", result.returncode)
    return False


def _write_sample_bars(data_root: Path, tickers: list[str]) -> None:
    store = ParquetBarStore(data_root)
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    base_index = pd.date_range(
        today + pd.Timedelta(hours=9, minutes=30),
        periods=390,
        freq="min",
        tz="America/New_York",
    )
    base_price = 100.0

    for ticker in tickers:
        prices = base_price + pd.Series(range(len(base_index))).mul(0.01)
        df = pd.DataFrame(
            {
                "timestamp": base_index,
                "open": prices,
                "high": prices + 0.05,
                "low": prices - 0.05,
                "close": prices + 0.02,
                "volume": 1_000,
            }
        )
        store.write_1m_bars(ticker.upper(), today.year, df)
        LOGGER.info("Wrote sample bars for %s (%d rows)", ticker.upper(), len(df))


def _write_sample_short_volume(short_root: Path) -> None:
    short_root.mkdir(parents=True, exist_ok=True)
    trade_date = date.today()
    out_path = short_root / f"{trade_date.strftime('%Y%m%d')}.parquet"
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(trade_date),
                "symbol": "SPY",
                "short_volume": 100_000,
                "short_exempt_volume": 2_000,
                "total_volume": 500_000,
                "market": "SAMPLE",
            }
        ]
    )
    df.to_parquet(out_path, index=False)
    LOGGER.info("Wrote sample short volume file: %s", out_path)


def _verify_data(data_root: Path, tickers: list[str], max_gap: int) -> bool:
    store = ParquetBarStore(data_root)
    all_ok = True
    for ticker in tickers:
        df = store.read_1m_bars(ticker.upper())
        if df.empty:
            LOGGER.warning("No bars found for %s", ticker.upper())
            all_ok = False
            continue
        ok = check_no_gap_more_than_n_bars(df, max_gap=max_gap)
        LOGGER.info(
            "Gap check %s for %s (rows=%d, max_gap=%d)",
            "PASS" if ok else "FAIL",
            ticker.upper(),
            len(df),
            max_gap,
        )
        all_ok = all_ok and ok
    return all_ok


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    _load_env()

    bars_root = REPO_ROOT / "data" / "bars"
    short_root = REPO_ROOT / "data" / "short_volume"
    bars_root.mkdir(parents=True, exist_ok=True)
    short_root.mkdir(parents=True, exist_ok=True)

    polygon_or_massive = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
    tradier_token = os.getenv("TRADIER_ACCESS_TOKEN")

    bars_has_data = _has_parquet_files(bars_root)
    short_has_data = _has_parquet_files(short_root)

    LOGGER.info("Bars data present: %s", bars_has_data)
    LOGGER.info("Short-volume data present: %s", short_has_data)

    if args.sample_only:
        LOGGER.info("--sample-only enabled; skipping API population")
    elif not (bars_has_data and short_has_data):
        if polygon_or_massive or tradier_token:
            phase2_ok = _run_phase2(args)
            if not phase2_ok:
                LOGGER.warning("Phase 2 command failed after retries")
        else:
            LOGGER.warning("No Polygon/Massive or Tradier key detected; API population skipped")

    bars_has_data = _has_parquet_files(bars_root)
    short_has_data = _has_parquet_files(short_root)

    if not args.skip_sample and (not bars_has_data or not short_has_data):
        LOGGER.info("Populating missing datasets with sample files")
        if not bars_has_data:
            _write_sample_bars(bars_root, args.tickers)
        if not short_has_data:
            _write_sample_short_volume(short_root)

    verified = _verify_data(bars_root, args.tickers, args.max_gap)
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
