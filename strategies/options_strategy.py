"""
Options Strategy — Lumibot strategy for regime-conditioned options trading.

Combines the full MFT engine with OptionsEngine (L1/L2/L3 structures),
GammaSurface (GEX features), ParameterTuner (adaptive thresholds), and
LearningCycleRunner (lesson feedback) into a single self-improving strategy.

Each trading iteration:
  1. Fetch OHLCV history → run RegimeEngine
  2. Fetch live options chain (yfinance fallback or Alpha Vantage)
  3. Fetch GEX features via GammaSurface
  4. Fetch L2 signals via PolygonRestClient (if key set)
  5. Run ParameterTuner.tune() → runtime thresholds
  6. Run OptionsEngine.select_trade() → OptionsTrade
  7. Execute multi-leg options order via Lumibot
  8. Record prediction in PredictionMixin + LearningCycleRunner lesson log

Backtest vs Live:
  - Lumibot's `is_backtesting` flag selects simulated order execution.
  - Multi-leg orders use Lumibot's `create_order()` with appropriate
    asset type and strike/expiry parameters.
  - In backtest mode, approximate P&L from mid-price changes.

Parameters (pass to Lumibot backtester or live runner)
-----------
  symbol            : underlying ticker (e.g. 'AAPL')
  lookback_bars     : OHLCV bars for RegimeEngine (default 300)
  min_confidence    : minimum trade confidence (default 0.40)
  max_open_trades   : max simultaneous multi-leg positions (default 3)
  auto_learn        : if True, run learning cycles after each N bars
  learn_every_n     : run learning cycle every N iterations (default 50)
  param_tuner_path  : path for persisting ParameterTuner banks (optional)

Requirements
-----------
  pip install lumibot yfinance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lazy Lumibot import (avoids credential checks at import time) ──────────────
try:
    from strategies._lumibot_lazy import Strategy, Asset, Order
    _LUMIBOT_OK = True
except ImportError:
    try:
        from lumibot.strategies import Strategy
        from lumibot.entities import Asset, Order
        _LUMIBOT_OK = True
    except ImportError:
        _LUMIBOT_OK = False
        Strategy = object  # type: ignore[assignment,misc]

from strategies.prediction_tracker import PredictionMixin


class OptionsStrategy(PredictionMixin, Strategy):  # type: ignore[misc]
    """
    Regime-conditioned multi-leg options strategy.

    Selects L1/L2/L3 options structures based on the full MFT engine,
    GEX surface, IV regime, and liquidity conditions.  Learns from each
    cycle's prediction errors via LearningCycleRunner.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Lumibot parameters (passed via parameters dict at strategy creation)
    # ──────────────────────────────────────────────────────────────────────
    parameters = {
        'symbol':           'SPY',
        'lookback_bars':    300,
        'min_confidence':   0.40,
        'max_open_trades':  3,
        'auto_learn':       True,
        'learn_every_n':    50,
        'param_tuner_path': '',
    }

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._iter_count   = 0
        self._open_trades: List[Dict[str, Any]] = []
        self._ohlcv_cache  = None
        self._chain_cache  = None

        self.sleeptime = '1D'  # daily iteration

        # Load config + build engine components
        from regime_engine.scanner import load_config, RegimeEngine
        from regime_engine.gamma_surface import GammaSurface
        from regime_engine.options_engine import OptionsEngine
        from regime_engine.auto_learning import LearningCycleRunner
        from regime_engine.param_tuner import ParameterTuner
        from regime_engine.variable_registry import VariableRegistry

        self._cfg = load_config()

        self._engine    = RegimeEngine(self._cfg)
        self._gamma     = GammaSurface(self._cfg.get('gamma', {}))
        self._options   = OptionsEngine(self._cfg.get('options_engine', {}))
        self._registry  = VariableRegistry(self._cfg.get('variable_registry', {}))
        self._learner   = LearningCycleRunner(
            self._cfg.get('auto_learning', {}),
            registry=self._registry,
        )
        self._tuner     = ParameterTuner(self._cfg.get('param_tuner', {}))

        # Load persisted tuner banks if path set
        tuner_path = self.parameters.get('param_tuner_path', '')
        if tuner_path:
            self._tuner.load(tuner_path)

        # Polygon REST for L2 (gracefully degrades if no key)
        from regime_engine.l2_feed import PolygonRestClient
        self._l2 = PolygonRestClient(config=self._cfg.get('polygon', {}))

        # AlphaVantage for options chain
        from regime_engine.data_fetcher import AlphaVantageFetcher
        self._av_fetcher = AlphaVantageFetcher(self._cfg)

        self._init_predictions()
        logger.info("OptionsStrategy initialized for %s", self.parameters['symbol'])

    def on_trading_iteration(self) -> None:
        self._iter_count += 1
        symbol       = self.parameters['symbol']
        lookback     = int(self.parameters.get('lookback_bars', 300))
        min_conf     = float(self.parameters.get('min_confidence', 0.40))
        max_trades   = int(self.parameters.get('max_open_trades', 3))

        # ── 1. Fetch OHLCV ────────────────────────────────────────────────
        import pandas as pd
        try:
            bars = self.get_historical_prices(symbol, lookback, 'day')
            if bars is None or bars.df is None or len(bars.df) < 50:
                return
            ohlcv = bars.df[['open', 'high', 'low', 'close', 'volume']].copy()
            ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
            self._ohlcv_cache = ohlcv
        except Exception as exc:
            logger.warning("OptionsStrategy: OHLCV fetch failed: %s", exc)
            return

        # ── 2. Run regime engine ──────────────────────────────────────────
        try:
            result = self._engine.run(ohlcv)
        except Exception as exc:
            logger.warning("OptionsStrategy: RegimeEngine.run failed: %s", exc)
            return

        if result is None:
            return

        regime_probs = dict(getattr(result, 'regime_probs', {}))
        composite    = float(getattr(result, 'composite_signal', 0.0))
        score        = float(getattr(result, 'score', 0.0))
        spot         = float(ohlcv['close'].iloc[-1])

        # ── 3. Fetch options chain (AV → yfinance fallback) ───────────────
        chain_df = self._fetch_chain(symbol)

        # ── 4. GEX features ───────────────────────────────────────────────
        gamma_features = {'gamma_wall_distance': 0.0, 'gamma_net': 0.0,
                          'gamma_expiry_days': 30.0, 'gex_flip_zone': 0.0}
        if chain_df is not None and not chain_df.empty:
            try:
                gamma_features = self._gamma.compute_features(chain_df, spot)
            except Exception as exc:
                logger.debug("GammaSurface failed: %s", exc)

        # ── 5. L2 signals ─────────────────────────────────────────────────
        l2_signals = None
        try:
            l2_signals = self._l2.get_snapshot(symbol)
        except Exception:
            pass

        # ── 6. ParameterTuner ─────────────────────────────────────────────
        overrides = self._tuner.tune(regime_probs, gamma_features, l2_signals)
        effective_threshold = float(overrides.get('signal_threshold', 0.20))
        effective_floor     = float(overrides.get('confidence_floor', 0.35))
        pos_mult            = float(overrides.get('position_size_mult', 1.0))

        # Apply tuner confidence floor on top of user min_confidence
        gate_conf = max(min_conf, effective_floor)

        # ── 7. Select options trade ───────────────────────────────────────
        hist_vol = self._compute_hist_vol(ohlcv)

        trade = None
        if chain_df is not None and not chain_df.empty:
            try:
                trade = self._options.select_trade(
                    regime_probs   = regime_probs,
                    gamma_features = gamma_features,
                    chain_df       = chain_df,
                    spot           = spot,
                    hist_vol_ann   = hist_vol,
                    l2_signals     = l2_signals,
                    min_confidence = gate_conf,
                )
            except Exception as exc:
                logger.warning("OptionsEngine.select_trade failed: %s", exc)

        # ── 8. Execute trade ──────────────────────────────────────────────
        if trade is not None and len(self._open_trades) < max_trades:
            self._execute_options_trade(trade, symbol, spot, pos_mult)
            self.record_options_prediction(
                symbol         = symbol,
                trade          = trade,
                spot           = spot,
                regime_probs   = regime_probs,
                gamma_features = gamma_features,
            )
            logger.info("OptionsStrategy: %s", trade.summary())
        else:
            # Directional equity fallback if no options available
            if abs(composite) >= effective_threshold and score >= gate_conf:
                self._execute_equity_trade(symbol, composite, score, pos_mult, spot)
            self.record_prediction(
                symbol = symbol,
                signal = 'UP' if composite > 0 else 'DOWN' if composite < 0 else 'NEUTRAL',
                price  = spot,
            )

        # ── 9. Auto-learning cycle ────────────────────────────────────────
        learn_every = int(self.parameters.get('learn_every_n', 50))
        if self.parameters.get('auto_learn', True) and self._iter_count % learn_every == 0:
            self._run_learning_cycle()

    # ──────────────────────────────────────────────────────────────────────
    # Trade execution helpers
    # ──────────────────────────────────────────────────────────────────────

    def _execute_options_trade(
        self,
        trade:    Any,   # OptionsTrade
        symbol:   str,
        spot:     float,
        pos_mult: float,
    ) -> None:
        """Submit multi-leg options orders through Lumibot."""
        portfolio_value = float(self.portfolio_value)
        risk_per_trade  = portfolio_value * 0.02 * pos_mult  # 2% risk × position multiplier

        for leg in trade.legs:
            if leg.mid_price <= 0:
                continue
            # Determine contract count from max risk
            cost_per_contract = leg.mid_price * 100
            if cost_per_contract <= 0:
                continue
            qty = max(1, int(risk_per_trade / (cost_per_contract * len(trade.legs) + 1)))

            try:
                asset = Asset(
                    symbol       = symbol,
                    asset_type   = 'option',
                    expiration   = leg.expiration,
                    strike       = leg.strike,
                    right        = leg.option_type.upper()[0],  # 'C' or 'P'
                    multiplier   = 100,
                )
                side = 'buy' if leg.action == 'buy' else 'sell'
                order = self.create_order(asset, qty, side, type='market')
                self.submit_order(order)
            except Exception as exc:
                logger.debug("OptionsStrategy: order submit failed for leg %s: %s", leg, exc)

        self._open_trades.append({
            'symbol':    symbol,
            'structure': trade.structure,
            'legs':      trade.legs,
            'entry_bar': self._iter_count,
            'spot':      spot,
            'trade':     trade,
        })

    def _execute_equity_trade(
        self,
        symbol:   str,
        signal:   float,
        score:    float,
        pos_mult: float,
        spot:     float,
    ) -> None:
        """Equity directional trade as fallback when no options available."""
        portfolio_value = float(self.portfolio_value)
        position_value  = portfolio_value * 0.10 * score * pos_mult
        shares          = max(1, int(position_value / (spot + 1e-10)))

        existing = self.get_position(symbol)

        if signal > 0:
            if existing is None or existing.quantity <= 0:
                try:
                    order = self.create_order(symbol, shares, 'buy', type='market')
                    self.submit_order(order)
                except Exception as exc:
                    logger.debug("equity order failed: %s", exc)
        elif signal < 0:
            if existing is not None and existing.quantity > 0:
                try:
                    order = self.create_order(symbol, existing.quantity, 'sell', type='market')
                    self.submit_order(order)
                except Exception as exc:
                    logger.debug("equity sell failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Options chain fetcher (AV → yfinance fallback)
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_chain(self, symbol: str):
        """Fetch options chain: try Alpha Vantage first, then yfinance."""
        # Try Alpha Vantage
        try:
            chain = self._av_fetcher.options_chain(symbol)
            if chain is not None and not chain.empty:
                return chain
        except Exception as exc:
            logger.debug("AV options_chain failed: %s", exc)

        # yfinance fallback
        try:
            return _yfinance_options_chain(symbol)
        except Exception as exc:
            logger.debug("yfinance options_chain failed: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hist_vol(ohlcv, window: int = 21) -> float:
        """Annualized realized volatility from log returns."""
        import numpy as np
        import math
        closes = ohlcv['close'].values
        if len(closes) < 2:
            return 0.20
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(log_returns) < 5:
            return 0.20
        recent = log_returns[-window:]
        return float(np.std(recent) * math.sqrt(252))

    def _run_learning_cycle(self) -> None:
        """Trigger a learning cycle and apply lessons to tuner."""
        if self._ohlcv_cache is None or len(self._ohlcv_cache) < 100:
            return
        try:
            result = self._learner.run_cycle(
                ohlcv_df    = self._ohlcv_cache,
                window_bars = min(250, len(self._ohlcv_cache)),
                stride_bars = 25,
            )
            scores = result.get('regime_scores', {})
            self._tuner.update_from_lessons(scores)
            logger.info(
                "OptionsStrategy: learning cycle complete | acc=%.1f%% | lessons=%d",
                result.get('cycle_accuracy', 0) * 100,
                result.get('n_lessons', 0),
            )
        except Exception as exc:
            logger.warning("OptionsStrategy: learning cycle failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Extended prediction tracking for options
    # ──────────────────────────────────────────────────────────────────────

    def record_options_prediction(
        self,
        symbol:         str,
        trade:          Any,   # OptionsTrade
        spot:           float,
        regime_probs:   Dict[str, float],
        gamma_features: Dict[str, float],
    ) -> None:
        """Record an options trade prediction for post-backtest scoring."""
        self._init_predictions()
        self._prediction_log.append({
            'date':          self.get_datetime(),
            'symbol':        symbol,
            'signal':        'OPTIONS',
            'price':         spot,
            'structure':     trade.structure,
            'level':         trade.level,
            'regime':        trade.regime,
            'vol_regime':    trade.vol_regime,
            'gex_regime':    trade.gex_regime,
            'confidence':    trade.confidence,
            'net_credit':    trade.net_credit,
            'max_profit':    trade.max_profit,
            'max_loss':      trade.max_loss,
            'breakeven':     trade.breakeven,
            'regime_probs':  regime_probs,
            'gamma_net':     gamma_features.get('gamma_net', 0.0),
            'gex_flip':      gamma_features.get('gex_flip_zone', 0.0),
            'legs':          [(l.option_type, l.action, l.strike, l.expiration, l.iv)
                              for l in trade.legs],
        })

    def on_backtest_end(self) -> None:
        """Score all predictions and log a summary after backtest completes."""
        from strategies.prediction_tracker import compute_prediction_accuracy
        scorecard = compute_prediction_accuracy(self)
        logger.info(
            "OptionsStrategy backtest complete | acc=%.1f%% | hits=%d/%d | edge=%.4f",
            scorecard.get('accuracy', 0) * 100,
            scorecard.get('hits', 0),
            scorecard.get('total_predictions', 0),
            scorecard.get('edge', 0),
        )

        # Persist learned parameter banks
        tuner_path = self.parameters.get('param_tuner_path', '')
        if tuner_path:
            self._tuner.save(tuner_path)


# ──────────────────────────────────────────────────────────────────────────────
# yfinance options chain helper
# ──────────────────────────────────────────────────────────────────────────────

def _yfinance_options_chain(symbol: str, max_exps: int = 4):
    """
    Fetch an options chain from yfinance and normalize to the same schema
    expected by GammaSurface and OptionsEngine.

    Returns a DataFrame with columns:
      strike, expiration, optiontype, openinterest, gamma,
      delta, impliedvolatility, volume, last, bid, ask, theta, vega
    """
    import yfinance as yf
    import pandas as pd

    tk = yf.Ticker(symbol)
    exps = tk.options
    if not exps:
        return pd.DataFrame()

    frames = []
    for exp in exps[:max_exps]:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        for df, opt_type in [(chain.calls, 'call'), (chain.puts, 'put')]:
            df = df.copy()
            df['optiontype']  = opt_type
            df['expiration']  = exp
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Normalize column names
    combined.columns = [c.lower().replace(' ', '').replace('_', '') for c in combined.columns]

    col_renames = {
        'impliedvolatility':   'impliedvolatility',
        'openinterest':        'openinterest',
        'contractsymbol':      'contractid',
        'lastprice':           'last',
        'inTheMoney':          'inthemoney',
    }
    combined.rename(columns={k: v for k, v in col_renames.items() if k in combined.columns}, inplace=True)

    # yfinance doesn't supply delta/gamma/theta/vega by default; set to 0
    for col in ('delta', 'gamma', 'theta', 'vega'):
        if col not in combined.columns:
            combined[col] = 0.0

    # Ensure volume → openinterest fallback
    if 'openinterest' not in combined.columns and 'volume' in combined.columns:
        combined['openinterest'] = combined['volume'].fillna(0)

    keep = ['strike', 'expiration', 'optiontype', 'openinterest',
            'impliedvolatility', 'delta', 'gamma', 'theta', 'vega',
            'bid', 'ask', 'last', 'volume']
    result = combined[[c for c in keep if c in combined.columns]].copy()

    return result
