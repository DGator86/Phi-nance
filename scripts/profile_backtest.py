#!/usr/bin/env python
"""Run a baseline profile against the vectorized backtest runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from phinance.backtest.runner import run_backtest
from phinance.utils.performance import PerformanceTracker, run_cprofile, track_time


def make_synthetic_ohlcv(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.04 / 252, 0.01, size=rows)
    close = 100 * np.exp(np.cumsum(drift))
    index = pd.date_range("2014-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, size=rows)),
            "high": close * (1 + rng.normal(0.002, 0.001, size=rows)),
            "low": close * (1 - rng.normal(0.002, 0.001, size=rows)),
            "close": close,
            "volume": rng.integers(200_000, 2_000_000, size=rows),
        },
        index=index,
    )


def run_baseline(rows: int, iterations: int, cprofile_out: str) -> None:
    tracker = PerformanceTracker()
    indicators = {
        "RSI": {"enabled": True, "params": {"period": 14}},
        "MACD": {"enabled": True, "params": {"fast": 12, "slow": 26, "signal": 9}},
    }

    with track_time(tracker, "prepare_dataset"):
        ohlcv = make_synthetic_ohlcv(rows)

    for i in range(iterations):
        with track_time(tracker, "run_backtest"):
            result = run_backtest(
                ohlcv=ohlcv,
                symbol="SPY",
                indicators=indicators,
                blend_weights={"RSI": 0.5, "MACD": 0.5},
                blend_method="weighted_sum",
                initial_capital=100_000,
            )
        print(
            f"iteration={i + 1}/{iterations} total_return={result.total_return:.4f} sharpe={result.sharpe:.4f} trades={result.total_trades}"
        )

    profile_text = run_cprofile(
        run_backtest,
        ohlcv=ohlcv,
        symbol="SPY",
        indicators=indicators,
        blend_weights={"RSI": 0.5, "MACD": 0.5},
        blend_method="weighted_sum",
        initial_capital=100_000,
        output_path=cprofile_out,
    )

    print("\n=== Timing Summary ===")
    print(tracker.as_markdown())
    print("\n=== cProfile Top Functions ===")
    print(profile_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2520, help="Number of business-day bars")
    parser.add_argument("--iterations", type=int, default=3, help="Backtest repeats for timing")
    parser.add_argument(
        "--cprofile-out",
        default="artifacts/perf/backtest_cprofile.txt",
        help="Output path for cProfile report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.cprofile_out).parent.mkdir(parents=True, exist_ok=True)
    run_baseline(rows=args.rows, iterations=args.iterations, cprofile_out=args.cprofile_out)
