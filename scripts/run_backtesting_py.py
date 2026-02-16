#!/usr/bin/env python3
"""
Run backtesting.py (kernc) backtest using Phi-nance bar data and optional projection strategy.

Requires: pip install phi-nance[backtesting]

  python -m scripts.run_backtesting_py
  python -m scripts.run_backtesting_py --strategy projection --ticker SPY --start 2024-01-01 --end 2024-06-30
  python -m scripts.run_backtesting_py --data-root data/bars

No broker or API keys needed. Uses same bar store as run_backtest.py (synthetic or data/bars).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="backtesting.py (kernc) backtest with Phi-nance data")
    p.add_argument("--ticker", default="SPY", help="Ticker symbol")
    p.add_argument("--start", default="2024-01-01", help="Start date")
    p.add_argument("--end", default="2024-06-30", help="End date")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bar store; if missing, use synthetic")
    p.add_argument("--strategy", choices=["sma_cross", "projection"], default="sma_cross", help="Strategy to run")
    p.add_argument("--commission", type=float, default=0.002, help="Commission per trade (fraction)")
    p.add_argument("--no-plot", action="store_true", help="Skip opening plot window")
    args = p.parse_args()

    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        print("backtesting.py is required. Install with: pip install phi-nance[backtesting]")
        return 1

    from phinence.backtesting_bridge.data import bar_store_to_bt_df
    from phinence.store.memory_store import InMemoryBarStore
    from phinence.store.parquet_store import ParquetBarStore
    from phinence.validation.backtest_runner import make_synthetic_bars

    if args.data_root.exists():
        bar_store = ParquetBarStore(args.data_root)
        if args.ticker.upper() not in [t.upper() for t in bar_store.list_tickers()]:
            print(f"Ticker {args.ticker} not in store; falling back to synthetic.")
            bar_store = InMemoryBarStore()
            bar_store.put_1m_bars(args.ticker, make_synthetic_bars(args.ticker, args.start, args.end, seed=hash(args.ticker) % 10000))
    else:
        bar_store = InMemoryBarStore()
        bar_store.put_1m_bars(args.ticker, make_synthetic_bars(args.ticker, args.start, args.end, seed=hash(args.ticker) % 10000))

    bt_df = bar_store_to_bt_df(bar_store, args.ticker, start=args.start, end=args.end, timeframe="1D")
    if bt_df.empty or len(bt_df) < 30:
        print("Not enough daily bars for backtest.")
        return 1

    if args.strategy == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
            import pandas as pd
            def SMA(values, n):
                return pd.Series(values).rolling(n).mean()

        class SmaCross(Strategy):
            n1 = 10
            n2 = 20

            def init(self):
                self.ma1 = self.I(SMA, self.data.Close, self.n1)
                self.ma2 = self.I(SMA, self.data.Close, self.n2)

            def next(self):
                if len(self.data) < self.n2:
                    return
                if crossover(self.ma1, self.ma2):
                    self.buy()
                elif crossover(self.ma2, self.ma1):
                    self.position.close()

        StrategyClass = SmaCross
    else:
        from phinence.assignment.engine import AssignmentEngine
        from phinence.composer.composer import Composer
        from phinence.engines.hedge import HedgeEngine
        from phinence.engines.liquidity import LiquidityEngine
        from phinence.engines.regime import RegimeEngine
        from phinence.engines.sentiment import SentimentEngine
        from phinence.backtesting_bridge.strategy import create_projection_strategy

        assigner = AssignmentEngine(bar_store)
        composer = Composer()
        engines = {
            "liquidity": LiquidityEngine(),
            "regime": RegimeEngine(),
            "sentiment": SentimentEngine(),
            "hedge": HedgeEngine(),
        }
        StrategyClass = create_projection_strategy(bar_store, assigner, engines, composer, args.ticker)

    bt = Backtest(bt_df, StrategyClass, commission=args.commission, exclusive_orders=True, trade_on_close=True)
    stats = bt.run()
    print(stats)
    if not args.no_plot:
        bt.plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
