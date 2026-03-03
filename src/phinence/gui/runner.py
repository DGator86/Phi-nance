"""
Run a single backtest by strategy name and return metrics dict for GUI display.
Uses backtesting.py (kernc) and Phi-nance bar store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]  # gui -> phinence -> src -> repo root

STRATEGY_CHOICES = [
    {"id": "buy_and_hold", "name": "Buy & Hold", "description": "Buy once and hold. Simple baseline."},
    {"id": "sma_cross", "name": "SMA Crossover", "description": "Buy when short average crosses above long; sell when it crosses below."},
    {"id": "projection", "name": "Phi-nance Projection", "description": "Uses our market projection (liquidity, regime, sentiment) for daily direction."},
]


def run_backtest_for_strategy(
    strategy_id: str,
    ticker: str,
    start: str,
    end: str,
    data_root: Path | None = None,
    commission: float = 0.002,
) -> dict[str, Any]:
    """
    Run one backtest for the given strategy. Returns a flat dict with keys like
    Strategy, Return %, Sharpe Ratio, Max Drawdown %, etc., for table display.
    """
    import pandas as pd
    data_root = data_root or REPO_ROOT / "data" / "bars"
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        return {"Strategy": strategy_id, "Error": "Install: pip install phi-nance[gui]"}

    from phinence.backtesting_bridge.data import bar_store_to_bt_df
    from phinence.store.memory_store import InMemoryBarStore
    from phinence.store.parquet_store import ParquetBarStore
    from phinence.validation.backtest_runner import make_synthetic_bars

    if data_root.exists():
        bar_store = ParquetBarStore(data_root)
        if ticker.upper() not in [t.upper() for t in bar_store.list_tickers()]:
            bar_store = InMemoryBarStore()
            bar_store.put_1m_bars(ticker, make_synthetic_bars(ticker, start, end, seed=hash(ticker) % 10000))
    else:
        bar_store = InMemoryBarStore()
        bar_store.put_1m_bars(ticker, make_synthetic_bars(ticker, start, end, seed=hash(ticker) % 10000))

    bt_df = bar_store_to_bt_df(bar_store, ticker, start=start, end=end, timeframe="1D")
    if bt_df.empty or len(bt_df) < 20:
        name = next((c["name"] for c in STRATEGY_CHOICES if c["id"] == strategy_id), strategy_id)
        return {"Strategy": name, "Error": "Not enough data for backtest"}

    if strategy_id == "buy_and_hold":
        class BuyAndHold(Strategy):
            def init(self):
                self._bought = False
            def next(self):
                if not self._bought:
                    self.buy()
                    self._bought = True
        StrategyClass = BuyAndHold
    elif strategy_id == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
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
    elif strategy_id == "projection":
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
        StrategyClass = create_projection_strategy(bar_store, assigner, engines, composer, ticker)
    else:
        return {"Strategy": strategy_id, "Error": "Unknown strategy"}

    bt = Backtest(bt_df, StrategyClass, commission=commission, exclusive_orders=True, trade_on_close=True, finalize_trades=True)
    stats = bt.run()
    name = next((c["name"] for c in STRATEGY_CHOICES if c["id"] == strategy_id), strategy_id)
    if not isinstance(stats, pd.Series):
        return {"Strategy": name, "Return [%]": 0, "Sharpe Ratio": 0, "Max. Drawdown [%]": 0}
    s = stats
    def _num(x, default=0):
        v = s.get(x, default)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)

    return {
        "Strategy": name,
        "Return [%]": round(_num("Return [%]"), 2),
        "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
        "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
        "# Trades": int(s.get("# Trades", 0) or 0),
        "Win Rate [%]": round(_num("Win Rate [%]"), 1),
    }
