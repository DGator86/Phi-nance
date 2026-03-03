#!/usr/bin/env python3
"""
Run Lumibot backtest using Phi-nance bar data (and optional projection pipeline).

Requires: pip install phi-nance[lumibot]

  python -m scripts.run_lumibot_backtest
  python -m scripts.run_lumibot_backtest --strategy projection --tickers SPY QQQ --start 2024-01-01 --end 2024-06-30
  python -m scripts.run_lumibot_backtest --data-root data/bars

With --strategy buy_and_hold: simple buy-on-first-day strategy (no projection).
With --strategy projection: uses assign -> engines -> MFM -> composer for daily direction (UP/DOWN) and trades.

Note: Lumibot may require TRADIER_TOKEN (or TRADIER_ACCESS_TOKEN in .env) to import; we use Pandas data only.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Lumibot 4.x loads Tradier broker on import and requires TRADIER_TOKEN (+ account/paper) to be set.
# We only use PandasDataBacktesting; if import fails, use Phi-nance native backtest: python -m scripts.run_backtest
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("TRADIER_TOKEN") and os.environ.get("TRADIER_ACCESS_TOKEN"):
        os.environ["TRADIER_TOKEN"] = os.environ["TRADIER_ACCESS_TOKEN"]
    # Lumibot Tradier broker also expects account_number and paper; set if missing so import can succeed
    if os.environ.get("TRADIER_TOKEN") and not os.environ.get("TRADIER_ACCOUNT_NUMBER"):
        os.environ.setdefault("TRADIER_ACCOUNT_NUMBER", "paper")
    if os.environ.get("TRADIER_TOKEN") and not os.environ.get("TRADIER_IS_PAPER"):
        os.environ.setdefault("TRADIER_IS_PAPER", "true")


def main() -> int:
    _load_env()
    p = argparse.ArgumentParser(description="Lumibot backtest with Phi-nance data (and optional projection strategy)")
    p.add_argument("--tickers", nargs="+", default=["SPY"], help="Tickers")
    p.add_argument("--start", default="2024-01-01", help="Start date")
    p.add_argument("--end", default="2024-06-30", help="End date")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bar store (Parquet); if missing, use synthetic")
    p.add_argument("--strategy", choices=["buy_and_hold", "projection"], default="buy_and_hold", help="Strategy to run")
    p.add_argument("--budget", type=float, default=100_000.0, help="Starting capital")
    args = p.parse_args()

    try:
        from lumibot.backtesting import PandasDataBacktesting
    except Exception as e:
        print("Lumibot could not be loaded:", e)
        print("Install with: pip install phi-nance[lumibot]")
        print("Lumibot 4.x may require TRADIER_TOKEN, TRADIER_ACCOUNT_NUMBER, TRADIER_IS_PAPER in .env.")
        print("For projection-accuracy backtest without Lumibot, use: python -m scripts.run_backtest")
        return 1

    from phinence.lumibot_bridge.data import bar_store_to_pandas_data
    from phinence.store.memory_store import InMemoryBarStore
    from phinence.store.parquet_store import ParquetBarStore
    from phinence.validation.backtest_runner import make_synthetic_bars

    # Build bar store
    if args.data_root.exists():
        bar_store = ParquetBarStore(args.data_root)
        tickers = [t for t in args.tickers if t.upper() in [x.upper() for x in bar_store.list_tickers()]] or args.tickers
    else:
        bar_store = InMemoryBarStore()
        for t in args.tickers:
            bar_store.put_1m_bars(t, make_synthetic_bars(t, args.start, args.end, seed=hash(t) % 10000))
        tickers = args.tickers

    # Build Lumibot pandas_data from our bar store
    try:
        pandas_data, datetime_start, datetime_end = bar_store_to_pandas_data(
            bar_store, tickers, args.start, args.end, timestep="minute"
        )
    except ValueError as e:
        print(e)
        return 1

    # Strategy: buy_and_hold or projection
    if args.strategy == "buy_and_hold":
        from lumibot.strategies import Strategy

        class BuyAndHold(Strategy):
            def initialize(self, parameters=None):
                self.sleeptime = "1D"

            def on_trading_iteration(self):
                if self.first_iteration and tickers:
                    sym = tickers[0]
                    order = self.create_order(sym, 100, "buy")
                    if order:
                        self.submit_order(order)

        StrategyClass = BuyAndHold
    else:
        from phinence.assignment.engine import AssignmentEngine
        from phinence.composer.composer import Composer
        from phinence.engines.hedge import HedgeEngine
        from phinence.engines.liquidity import LiquidityEngine
        from phinence.engines.regime import RegimeEngine
        from phinence.engines.sentiment import SentimentEngine
        from phinence.lumibot_bridge.strategy import create_projection_strategy_class

        assigner = AssignmentEngine(bar_store)
        composer = Composer()
        engines = {
            "liquidity": LiquidityEngine(),
            "regime": RegimeEngine(),
            "sentiment": SentimentEngine(),
            "hedge": HedgeEngine(),
        }
        StrategyClass = create_projection_strategy_class(
            bar_store, assigner, engines, composer, tickers
        )

    # Run backtest
    print(f"Running Lumibot backtest: {args.strategy} on {tickers} from {datetime_start} to {datetime_end}")
    result = StrategyClass.run_backtest(
        PandasDataBacktesting,
        datetime_start.to_pydatetime() if hasattr(datetime_start, "to_pydatetime") else datetime_start,
        datetime_end.to_pydatetime() if hasattr(datetime_end, "to_pydatetime") else datetime_end,
        pandas_data=pandas_data,
        budget=args.budget,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_progress_bar=True,
    )
    print("Backtest finished.")
    if result is not None and hasattr(result, "stats"):
        print(result.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
