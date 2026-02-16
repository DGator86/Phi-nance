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
    {"id": "rsi", "name": "RSI Strategy", "description": "RSI oversold/overbought."},
    {"id": "bollinger", "name": "Bollinger Bands", "description": "BB mean reversion."},
    {"id": "macd", "name": "MACD", "description": "MACD crossover."},
    {"id": "momentum", "name": "Momentum", "description": "Price momentum."},
    {"id": "mean_reversion", "name": "Mean Reversion", "description": "SMA mean reversion."},
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
    strategy_params: dict[str, Any] | None = None,
    testing_mode: str | None = None,
    trade_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run one backtest for the given strategy. Returns a flat dict with keys like
    Strategy, Return %, Sharpe Ratio, Max Drawdown %, etc., for table display.

    testing_mode: "phi_mode" (technical/Phi metrics only) or "trade_mode" (full P&L with account/costs/PDT).
    trade_config: for trade_mode: initial_balance, commission_pct, commission_per_trade, brokerage, pdt_effect.
    """
    import pandas as pd
    data_root = data_root or REPO_ROOT / "data" / "bars"
    trade_config = trade_config or {}
    testing_mode = testing_mode or "trade_mode"
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

    strategy_params = strategy_params or {}
    
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
        fast_period = strategy_params.get("fast_period", 10)
        slow_period = strategy_params.get("slow_period", 20)
        class SmaCross(Strategy):
            def init(self):
                self.ma1 = self.I(SMA, self.data.Close, fast_period)
                self.ma2 = self.I(SMA, self.data.Close, slow_period)
            def next(self):
                if len(self.data) < slow_period:
                    return
                if crossover(self.ma1, self.ma2):
                    self.buy()
                elif crossover(self.ma2, self.ma1):
                    self.position.close()
        StrategyClass = SmaCross
    elif strategy_id == "rsi":
        rsi_period = int(strategy_params.get("rsi_period", 14))
        oversold = float(strategy_params.get("oversold", 30.0))
        overbought = float(strategy_params.get("overbought", 70.0))
        class RSIStrategy(Strategy):
            def init(self):
                def RSI(values, period):
                    delta = pd.Series(values).diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))
                self.rsi = self.I(RSI, self.data.Close, rsi_period)
            def next(self):
                if len(self.data) < rsi_period:
                    return
                if self.rsi[-1] < oversold and not self.position:
                    self.buy()
                elif self.rsi[-1] > overbought and self.position:
                    self.position.close()
        StrategyClass = RSIStrategy
    elif strategy_id == "bollinger":
        bb_period = int(strategy_params.get("bb_period", 20))
        num_std = float(strategy_params.get("num_std", 2.0))
        class BollingerStrategy(Strategy):
            def init(self):
                def BB(values, period, std):
                    s = pd.Series(values)
                    sma = s.rolling(period).mean()
                    std_dev = s.rolling(period).std()
                    return sma, sma + (std_dev * std), sma - (std_dev * std)
                self.bb_mid, self.bb_upper, self.bb_lower = self.I(BB, self.data.Close, bb_period, num_std)
            def next(self):
                if len(self.data) < bb_period:
                    return
                if self.data.Close[-1] <= self.bb_lower[-1] and not self.position:
                    self.buy()
                elif self.data.Close[-1] >= self.bb_upper[-1] and self.position:
                    self.position.close()
        StrategyClass = BollingerStrategy
    elif strategy_id == "macd":
        fast_period = int(strategy_params.get("fast_period", 12))
        slow_period = int(strategy_params.get("slow_period", 26))
        signal_period = int(strategy_params.get("signal_period", 9))
        class MACDStrategy(Strategy):
            def init(self):
                def EMA(values, period):
                    return pd.Series(values).ewm(span=period, adjust=False).mean()
                def MACD(values, fast, slow, signal):
                    ema_fast = EMA(values, fast)
                    ema_slow = EMA(values, slow)
                    macd_line = ema_fast - ema_slow
                    signal_line = EMA(macd_line, signal)
                    return macd_line, signal_line
                self.macd, self.signal = self.I(MACD, self.data.Close, fast_period, slow_period, signal_period)
            def next(self):
                if len(self.data) < slow_period + signal_period:
                    return
                if crossover(self.macd, self.signal) and not self.position:
                    self.buy()
                elif crossover(self.signal, self.macd) and self.position:
                    self.position.close()
        StrategyClass = MACDStrategy
    elif strategy_id == "momentum":
        lookback = int(strategy_params.get("lookback", 20))
        class MomentumStrategy(Strategy):
            def init(self):
                pass
            def next(self):
                if len(self.data) < lookback + 1:
                    return
                momentum = (self.data.Close[-1] - self.data.Close[-lookback]) / self.data.Close[-lookback]
                if momentum > 0 and not self.position:
                    self.buy()
                elif momentum < 0 and self.position:
                    self.position.close()
        StrategyClass = MomentumStrategy
    elif strategy_id == "mean_reversion":
        sma_period = int(strategy_params.get("sma_period", 20))
        class MeanReversionStrategy(Strategy):
            def init(self):
                try:
                    from backtesting.test import SMA
                except ImportError:
                    def SMA(values, n):
                        return pd.Series(values).rolling(n).mean()
                self.sma = self.I(SMA, self.data.Close, sma_period)
            def next(self):
                if len(self.data) < sma_period:
                    return
                if self.data.Close[-1] < self.sma[-1] and not self.position:
                    self.buy()
                elif self.data.Close[-1] > self.sma[-1] and self.position:
                    self.position.close()
        StrategyClass = MeanReversionStrategy
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

    # Trade mode: initial cash and commission from trade_config; Phi mode: defaults
    initial_cash = float(trade_config.get("initial_balance", 10000)) if testing_mode == "trade_mode" else 10000
    if testing_mode == "trade_mode":
        comm_pct = float(trade_config.get("commission_pct", 0.1)) / 100.0
        comm_fixed = float(trade_config.get("commission_per_trade", 0.0))
        commission = (comm_fixed, comm_pct) if comm_fixed else comm_pct
    bt = Backtest(
        bt_df, StrategyClass,
        cash=initial_cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True,
        finalize_trades=True,
    )
    stats = bt.run()
    name = next((c["name"] for c in STRATEGY_CHOICES if c["id"] == strategy_id), strategy_id)
    if not isinstance(stats, pd.Series):
        out = {"Strategy": name, "Return [%]": 0, "Sharpe Ratio": 0, "Max. Drawdown [%]": 0}
    else:
        s = stats
        def _num(x, default=0):
            v = s.get(x, default)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return default
            return float(v)
        out = {
            "Strategy": name,
            "Return [%]": round(_num("Return [%]"), 2),
            "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
            "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
            "# Trades": int(s.get("# Trades", 0) or 0),
            "Win Rate [%]": round(_num("Win Rate [%]"), 1),
        }
        if testing_mode == "trade_mode":
            out["Initial Balance"] = initial_cash
            out["Brokerage"] = trade_config.get("brokerage", "—")
            if trade_config.get("pdt_effect"):
                n_trades = int(s.get("# Trades", 0) or 0)
                out["PDT"] = "⚠️ On (≤3 day trades/5d)" if n_trades > 3 else "On"
        if testing_mode == "phi_mode":
            # Phi/technical metrics: direction accuracy (sign of return vs signal)
            returns = bt_df["Close"].pct_change().dropna()
            if len(returns) > 0:
                # Simple proxy: strategy would have been long; compare to actual return
                dir_correct = (returns > 0).sum() / len(returns) if len(returns) else 0.5
                out["φ Direction Accuracy"] = round(float(dir_correct) * 100, 1)
            out["φ Mode"] = "Technical"
    return out


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


def _get_strategy_signal(strategy_id: str, bt_df: pd.DataFrame, bar_store: Any, ticker: str, idx: int, params: dict[str, Any] | None = None) -> int:
    """
    Get signal from a strategy at a given index without executing trades.
    Returns: 1 = buy, -1 = sell, 0 = hold/neutral
    """
    from backtesting.lib import crossover
    import pandas as pd
    
    if strategy_id == "buy_and_hold":
        # Buy on first bar only
        return 1 if idx == 0 else 0
    
    params = params or {}
    
    elif strategy_id == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
            def SMA(values, n):
                return pd.Series(values).rolling(n).mean()
        
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 20)
        
        if idx < slow_period:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        ma1 = SMA(closes, fast_period)
        ma2 = SMA(closes, slow_period)
        
        if len(ma1) < 2 or len(ma2) < 2:
            return 0
        
        if ma1.iloc[-2] <= ma2.iloc[-2] and ma1.iloc[-1] > ma2.iloc[-1]:
            return 1
        elif ma1.iloc[-2] >= ma2.iloc[-2] and ma1.iloc[-1] < ma2.iloc[-1]:
            return -1
        return 0
    elif strategy_id == "rsi":
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        
        if idx < rsi_period:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) < 1:
            return 0
        
        if rsi.iloc[-1] < oversold:
            return 1
        elif rsi.iloc[-1] > overbought:
            return -1
        return 0
    elif strategy_id == "bollinger":
        bb_period = int(params.get("bb_period", 20))
        num_std = float(params.get("num_std", 2.0))
        
        if idx < bb_period:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        sma = closes.rolling(bb_period).mean()
        std = closes.rolling(bb_period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        if len(closes) < 1:
            return 0
        
        if closes.iloc[-1] <= lower.iloc[-1]:
            return 1
        elif closes.iloc[-1] >= upper.iloc[-1]:
            return -1
        return 0
    elif strategy_id == "macd":
        fast_period = int(params.get("fast_period", 12))
        slow_period = int(params.get("slow_period", 26))
        signal_period = int(params.get("signal_period", 9))
        
        if idx < slow_period + signal_period:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return 0
        
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return 1
        elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return -1
        return 0
    elif strategy_id == "momentum":
        lookback = int(params.get("lookback", 20))
        
        if idx < lookback:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        momentum = (closes.iloc[-1] - closes.iloc[-lookback]) / closes.iloc[-lookback]
        
        if momentum > 0:
            return 1
        elif momentum < 0:
            return -1
        return 0
    elif strategy_id == "mean_reversion":
        sma_period = int(params.get("sma_period", 20))
        
        if idx < sma_period:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        sma = closes.rolling(sma_period).mean()
        
        if len(closes) < 1 or len(sma) < 1:
            return 0
        
        if closes.iloc[-1] < sma.iloc[-1]:
            return 1
        elif closes.iloc[-1] > sma.iloc[-1]:
            return -1
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


def detect_regime(
    ticker: str,
    start: str,
    end: str,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Detect market regime for the given period using RegimeEngine.
    Returns regime probabilities and best regime.
    """
    import pandas as pd
    from datetime import datetime, timezone
    
    data_root = data_root or REPO_ROOT / "data" / "bars"
    from phinence.backtesting_bridge.data import bar_store_to_bt_df
    from phinence.store.memory_store import InMemoryBarStore
    from phinence.store.parquet_store import ParquetBarStore
    from phinence.validation.backtest_runner import make_synthetic_bars
    from phinence.assignment.engine import AssignmentEngine
    from phinence.engines.regime import RegimeEngine
    
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
        return {
            "best_regime": "Unknown",
            "regime_distribution": {"Trending": 0.33, "Mean Reverting": 0.33, "Expanding": 0.34},
            "alignment": "neutral"
        }
    
    # Get bars for regime analysis
    bars_1m = bar_store.read_1m_bars(ticker)
    if bars_1m.empty:
        bars_1m = pd.DataFrame({
            "timestamp": bt_df.index,
            "open": bt_df["Open"],
            "high": bt_df["High"],
            "low": bt_df["Low"],
            "close": bt_df["Close"],
            "volume": bt_df["Volume"],
        })
    
    # Sample regime at multiple points
    assigner = AssignmentEngine(bar_store)
    regime_engine = RegimeEngine()
    
    regime_samples = []
    sample_points = min(10, len(bt_df))
    step = max(1, len(bt_df) // sample_points)
    
    for i in range(0, len(bt_df), step):
        bar_date = bt_df.index[i]
        if isinstance(bar_date, pd.Timestamp):
            as_of = bar_date.to_pydatetime()
            start_ts = bt_df.index[0]
            end_ts = bar_date
        else:
            as_of = datetime.now(timezone.utc)
            start_ts = pd.Timestamp(bt_df.index[0])
            end_ts = pd.Timestamp(bar_date)
        
        try:
            packet = assigner.assign(ticker, as_of, start_ts=start_ts, end_ts=end_ts)
            regime_result = regime_engine.run(packet)
            if regime_result.get("regime_probs"):
                regime_samples.append(regime_result["regime_probs"])
        except Exception:
            continue
    
    if not regime_samples:
        return {
            "best_regime": "Unknown",
            "regime_distribution": {"Trending": 0.33, "Mean Reverting": 0.33, "Expanding": 0.34},
            "alignment": "neutral"
        }
    
    # Average regime probabilities
    avg_probs = {
        "trend": sum(s.get("trend", 0) for s in regime_samples) / len(regime_samples),
        "mean_revert": sum(s.get("mean_revert", 0) for s in regime_samples) / len(regime_samples),
        "expansion": sum(s.get("expansion", 0) for s in regime_samples) / len(regime_samples),
    }
    
    # Determine best regime
    best_regime = max(avg_probs.items(), key=lambda x: x[1])[0]
    regime_names = {
        "trend": "Trending Up",
        "mean_revert": "Mean Reverting",
        "expansion": "Expanding Volatility"
    }
    
    return {
        "best_regime": regime_names.get(best_regime, "Unknown"),
        "regime_distribution": {
            "Trending": avg_probs["trend"],
            "Mean Reverting": avg_probs["mean_revert"],
            "Expanding": avg_probs["expansion"]
        },
        "alignment": regime_samples[-1].get("alignment", "neutral") if regime_samples else "neutral"
    }


def run_combined_backtest(
    strategy_ids: list[str],
    ticker: str,
    start: str,
    end: str,
    voting_mode: str = "majority",
    data_root: Path | None = None,
    commission: float = 0.002,
    strategy_params_map: dict[str, dict[str, Any]] | None = None,
    compounding_params: dict[str, Any] | None = None,
    testing_mode: str | None = None,
    trade_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a backtest combining multiple strategies using voting logic.
    testing_mode and trade_config: same as run_backtest_for_strategy.
    """
    import pandas as pd
    trade_config = trade_config or {}
    testing_mode = testing_mode or "trade_mode"
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
    
    # Pre-compute signals for all strategies (with parameters)
    signals_by_strategy = {}
    strategy_params_map = strategy_params_map or {}
    compounding_params = compounding_params or {}
    
    for sid in strategy_ids:
        params = strategy_params_map.get(sid, {})
        signals = []
        for idx in range(len(bt_df)):
            try:
                signal = _get_strategy_signal(sid, bt_df, bar_store, ticker, idx, params)
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
                # Weighted: use thresholds from compounding_params
                buy_threshold = compounding_params.get("buy_threshold", 0.3) if compounding_params else 0.3
                sell_threshold = compounding_params.get("sell_threshold", -0.3) if compounding_params else -0.3
                weighted_signal = sum(signals) / len(signals) if signals else 0
                if weighted_signal > buy_threshold:
                    if not self.position:
                        self.buy()
                elif weighted_signal < sell_threshold:
                    if self.position:
                        self.position.close()
            elif voting_mode == "at_least_n":
                min_agreement = compounding_params.get("min_agreement", 2) if compounding_params else 2
                buy_votes = sum(1 for s in signals if s > 0)
                sell_votes = sum(1 for s in signals if s < 0)
                if buy_votes >= min_agreement and not self.position:
                    self.buy()
                elif sell_votes >= min_agreement and self.position:
                    self.position.close()
    
    # Trade mode: cash and commission
    initial_cash = float(trade_config.get("initial_balance", 10000)) if testing_mode == "trade_mode" else 10000
    if testing_mode == "trade_mode":
        comm_pct = float(trade_config.get("commission_pct", 0.1)) / 100.0
        comm_fixed = float(trade_config.get("commission_per_trade", 0.0))
        commission = (comm_fixed, comm_pct) if comm_fixed else comm_pct
    bt = Backtest(
        bt_df, CombinedStrategy,
        cash=initial_cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True,
        finalize_trades=True,
    )
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
    
    out = {
        "Strategy": combined_name,
        "Return [%]": round(_num("Return [%]"), 2),
        "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
        "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
        "# Trades": int(s.get("# Trades", 0) or 0),
        "Win Rate [%]": round(_num("Win Rate [%]"), 1),
    }
    if testing_mode == "trade_mode":
        out["Initial Balance"] = initial_cash
        out["Brokerage"] = trade_config.get("brokerage", "—")
        if trade_config.get("pdt_effect"):
            n_trades = int(s.get("# Trades", 0) or 0)
            out["PDT"] = "⚠️ On (≤3 day trades/5d)" if n_trades > 3 else "On"
    if testing_mode == "phi_mode":
        out["φ Mode"] = "Technical"
    return out
