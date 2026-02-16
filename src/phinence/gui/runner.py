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

# Voting modes for combining strategies
VOTING_MODES = [
    {"id": "majority", "name": "Majority Vote", "description": "Buy/sell when most strategies agree"},
    {"id": "unanimous", "name": "Unanimous", "description": "Buy/sell only when all strategies agree"},
    {"id": "weighted", "name": "Weighted Average", "description": "Weight signals by strategy performance"},
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


def _create_single_strategy_instance(strategy_id: str, bt_df: pd.DataFrame, bar_store: Any, ticker: str) -> type:
    """Create a Strategy class for a single strategy ID."""
    from backtesting import Strategy
    from backtesting.lib import crossover
    import pandas as pd
    
    if strategy_id == "buy_and_hold":
        class BuyAndHold(Strategy):
            def init(self):
                self._bought = False
            def next(self):
                if not self._bought:
                    self.buy()
                    self._bought = True
        return BuyAndHold
    
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
        return SmaCross
    
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
        return create_projection_strategy(bar_store, assigner, engines, composer, ticker)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_id}")


def _get_strategy_signal(strategy_id: str, bt_df: pd.DataFrame, bar_store: Any, ticker: str, idx: int) -> int:
    """
    Get signal from a strategy at a given index without executing trades.
    Returns: 1 = buy, -1 = sell, 0 = hold/neutral
    """
    from backtesting.lib import crossover
    import pandas as pd
    
    if strategy_id == "buy_and_hold":
        # Buy on first bar only
        return 1 if idx == 0 else 0
    
    elif strategy_id == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
            def SMA(values, n):
                return pd.Series(values).rolling(n).mean()
        
        if idx < 20:  # Need at least 20 bars for SMA(20)
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        ma1 = SMA(closes, 10)
        ma2 = SMA(closes, 20)
        
        if len(ma1) < 2 or len(ma2) < 2:
            return 0
        
        # Check crossover
        if ma1.iloc[-2] <= ma2.iloc[-2] and ma1.iloc[-1] > ma2.iloc[-1]:
            return 1  # Golden cross - buy
        elif ma1.iloc[-2] >= ma2.iloc[-2] and ma1.iloc[-1] < ma2.iloc[-1]:
            return -1  # Death cross - sell
        return 0
    
    elif strategy_id == "projection":
        # For projection, we'd need to run the full pipeline
        # For now, return neutral - this would need more complex integration
        try:
            from phinence.assignment.engine import AssignmentEngine
            from phinence.composer.composer import Composer
            from phinence.engines.hedge import HedgeEngine
            from phinence.engines.liquidity import LiquidityEngine
            from phinence.engines.regime import RegimeEngine
            from phinence.engines.sentiment import SentimentEngine
            from phinence.mfm.merger import build_mfm
            from datetime import datetime, timezone
            
            if idx == 0:
                return 0
            
            # Get date for this bar
            bar_date = bt_df.index[idx]
            if isinstance(bar_date, pd.Timestamp):
                as_of = bar_date.to_pydatetime()
                start_ts = bt_df.index[0]
                end_ts = bar_date
            else:
                as_of = datetime.now(timezone.utc)
                start_ts = pd.Timestamp(bt_df.index[0])
                end_ts = pd.Timestamp(bar_date)
            
            assigner = AssignmentEngine(bar_store)
            composer = Composer()
            engines = {
                "liquidity": LiquidityEngine(),
                "regime": RegimeEngine(),
                "sentiment": SentimentEngine(),
                "hedge": HedgeEngine(),
            }
            packet = assigner.assign(ticker, as_of, start_ts=start_ts, end_ts=end_ts)
            
            liq = engines["liquidity"].run(packet)
            reg = engines["regime"].run(packet)
            sent = engines["sentiment"].run(packet)
            hed = engines["hedge"].run(packet)
            
            mfm = build_mfm(ticker, as_of, liquidity=liq, regime=reg, sentiment=sent, hedge=hed)
            proj = composer.run(mfm, horizons=["1d"])
            
            hp = proj.get_horizon("1d")
            if hp and hp.direction.up > 0.55:
                return 1
            elif hp and hp.direction.up < 0.45:
                return -1
            return 0
        except Exception:
            return 0
    
    return 0


def run_combined_backtest(
    strategy_ids: list[str],
    ticker: str,
    start: str,
    end: str,
    voting_mode: str = "majority",
    data_root: Path | None = None,
    commission: float = 0.002,
) -> dict[str, Any]:
    """
    Run a backtest combining multiple strategies using voting logic.
    
    Args:
        strategy_ids: List of strategy IDs to combine
        voting_mode: "majority", "unanimous", or "weighted"
    
    Returns:
        Dict with metrics for the combined strategy
    """
    import pandas as pd
    data_root = data_root or REPO_ROOT / "data" / "bars"
    try:
        from backtesting import Backtest, Strategy
    except ImportError:
        return {"Strategy": "+".join(strategy_ids), "Error": "Install: pip install phi-nance[gui]"}
    
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
        return {"Strategy": "+".join(strategy_ids), "Error": "Not enough data for backtest"}
    
    # Pre-compute signals for all strategies
    signals_by_strategy = {}
    for sid in strategy_ids:
        signals = []
        for idx in range(len(bt_df)):
            try:
                signal = _get_strategy_signal(sid, bt_df, bar_store, ticker, idx)
                signals.append(signal)
            except Exception:
                signals.append(0)
        signals_by_strategy[sid] = signals
    
    # Create combined strategy class
    class CombinedStrategy(Strategy):
        def init(self):
            pass
        
        def next(self):
            idx = len(self.data) - 1
            if idx < 0:
                return
            
            # Get signals from all strategies at this index
            signals = []
            for sid in strategy_ids:
                if sid in signals_by_strategy and idx < len(signals_by_strategy[sid]):
                    signals.append(signals_by_strategy[sid][idx])
                else:
                    signals.append(0)
            
            # Combine signals based on voting mode
            if voting_mode == "majority":
                buy_votes = sum(1 for s in signals if s > 0)
                sell_votes = sum(1 for s in signals if s < 0)
                if buy_votes > sell_votes and buy_votes > len(signals) / 2:
                    if not self.position:
                        self.buy()
                elif sell_votes > buy_votes and sell_votes > len(signals) / 2:
                    if self.position:
                        self.position.close()
            
            elif voting_mode == "unanimous":
                all_buy = all(s > 0 for s in signals)
                all_sell = all(s < 0 for s in signals)
                if all_buy and not self.position:
                    self.buy()
                elif all_sell and self.position:
                    self.position.close()
            
            elif voting_mode == "weighted":
                # Simple weighted: average signals
                weighted_signal = sum(signals) / len(signals) if signals else 0
                if weighted_signal > 0.3:  # Threshold for buy
                    if not self.position:
                        self.buy()
                elif weighted_signal < -0.3:  # Threshold for sell
                    if self.position:
                        self.position.close()
    
    # Run backtest
    bt = Backtest(bt_df, CombinedStrategy, commission=commission, exclusive_orders=True, trade_on_close=True, finalize_trades=True)
    stats = bt.run()
    
    strategy_names = [next((c["name"] for c in STRATEGY_CHOICES if c["id"] == sid), sid) for sid in strategy_ids]
    combined_name = " + ".join(strategy_names)
    
    if not isinstance(stats, pd.Series):
        return {"Strategy": combined_name, "Return [%]": 0, "Sharpe Ratio": 0, "Max. Drawdown [%]": 0}
    
    s = stats
    def _num(x, default=0):
        v = s.get(x, default)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    
    return {
        "Strategy": combined_name,
        "Return [%]": round(_num("Return [%]"), 2),
        "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
        "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
        "# Trades": int(s.get("# Trades", 0) or 0),
        "Win Rate [%]": round(_num("Win Rate [%]"), 1),
    }
