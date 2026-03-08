#!/usr/bin/env python3
"""Auto-training script for PhiAI models."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phi.data.cache import fetch_and_cache
from phi.logging import get_logger
from phi.phiai.auto_tune import run_phiai_optimization, save_best_params
from phi.utils.validation import sanitize_ticker

logger = get_logger(__name__)

DEFAULT_INDICATORS: dict[str, dict[str, object]] = {
    "RSI": {"enabled": True, "auto_tune": True, "params": {}},
    "MACD": {"enabled": True, "auto_tune": True, "params": {}},
    "BBands": {"enabled": True, "auto_tune": True, "params": {}},
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for auto-training."""
    parser = argparse.ArgumentParser(description="Auto-train PhiAI models.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to train on (space-separated)")
    parser.add_argument("--timeframe", default="1D", help="Data timeframe (e.g., 1D, 1H)")
    parser.add_argument("--years", type=int, default=3, help="Years of historical data to use")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--windows", type=int, default=3, help="Number of walk-forward windows")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--metric", default="sharpe", help="Optimization metric (sharpe, roi, etc.)")
    parser.add_argument("--output-dir", default="./runs/best_params", help="Directory to save best parameters")
    parser.add_argument("--vendor", default="yfinance", help="Data vendor")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh data even if cached")
    parser.add_argument("--verbose", action="store_true", help="Increase log level to DEBUG")
    return parser.parse_args()


def main() -> None:
    """Run auto-training for each requested ticker."""
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date() if not args.end_date else datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=365 * args.years)

    logger.info("Starting auto-training for %s from %s to %s", args.tickers, start_date, end_date)

    processed = 0
    saved = 0
    skipped = 0

    for raw_ticker in args.tickers:
        processed += 1
        try:
            ticker = sanitize_ticker(raw_ticker)
            logger.info("Processing %s", ticker)

            data = fetch_and_cache(
                vendor=args.vendor,
                symbol=ticker,
                timeframe=args.timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                force_refresh=args.force_refresh,
            )
            if data.empty:
                skipped += 1
                logger.warning("No data for %s, skipping", ticker)
                continue

            result = run_phiai_optimization(
                ohlcv=data,
                indicators_config=DEFAULT_INDICATORS,
                n_trials=args.n_trials,
                walk_forward_windows=args.windows,
                parallel_jobs=args.parallel,
                metric=args.metric,
            )

            dataset_id = f"{ticker}_{args.timeframe}_{start_date}_{end_date}"
            save_best_params(
                result["best_params"],
                dataset_id=dataset_id,
                metric=args.metric,
                best_value=float(result.get("best_value", 0.0)),
                output_dir=output_path,
            )

            saved += 1
            logger.info("Best params for %s: %s", ticker, result.get("explanation", "(no explanation)"))
        except Exception:
            skipped += 1
            logger.exception("Failed to process %s", raw_ticker)

    logger.info("Auto-training completed. processed=%d saved=%d skipped=%d", processed, saved, skipped)


if __name__ == "__main__":
    main()
