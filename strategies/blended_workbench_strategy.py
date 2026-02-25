"""
BlendedWorkbenchStrategy
------------------------
Lumibot strategy that computes multiple indicator signals from OHLCV,
blends them via phi.blending, and trades on the composite signal.

Uses phi.indicators.simple for signal computation.
Supports Regime-Weighted blending via regime_engine when blend_method is regime_weighted.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

os.environ["IS_BACKTESTING"] = "True"

import pandas as pd

from lumibot.strategies import Strategy

from strategies.prediction_tracker import PredictionMixin


class BlendedWorkbenchStrategy(PredictionMixin, Strategy):
    """
    Multi-indicator blended strategy for the Live Backtest Workbench.
    """

    parameters = {
        "symbol": "SPY",
        "indicators": {},      # {name: {params}}
        "blend_method": "weighted_sum",
        "blend_weights": {},
        "signal_threshold": 0.15,
        "lookback_bars": 200,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()
        self._engine = None

    def _get_regime_engine(self):
        if self._engine is None:
            import yaml
            cfg_path = os.path.join("regime_engine", "config.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            from regime_engine.scanner import RegimeEngine
            self._engine = RegimeEngine(cfg)
        return self._engine

    def on_trading_iteration(self):
        symbol = str(self.parameters.get("symbol", "SPY"))
        indicators = dict(self.parameters.get("indicators", {}))
        blend_method = str(self.parameters.get("blend_method", "weighted_sum"))
        blend_weights = dict(self.parameters.get("blend_weights", {}))
        threshold = float(self.parameters.get("signal_threshold", 0.15))
        lookback = int(self.parameters.get("lookback_bars", 200))

        bars = self.get_historical_prices(symbol, lookback, "day")
        if bars is None or bars.df is None or len(bars.df) < 50:
            return

        df = bars.df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return

        from phi.indicators.simple import compute_indicator, INDICATOR_COMPUTERS

        signals_dict = {}
        for name, cfg in indicators.items():
            if name not in INDICATOR_COMPUTERS:
                continue
            params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
            try:
                sig = compute_indicator(name, df, params)
                if sig is not None and not sig.empty:
                    signals_dict[name] = sig
            except Exception:
                pass

        if not signals_dict:
            return

        signals_df = pd.DataFrame(signals_dict)
        signals_df = signals_df.reindex(df.index).ffill().bfill()

        regime_probs = None
        if blend_method == "regime_weighted":
            try:
                engine = self._get_regime_engine()
                out = engine.run(df)
                regime_probs = out.get("regime_probs")
            except Exception:
                regime_probs = None
                blend_method = "weighted_sum"

        from phi.blending import blend_signals
        composite = blend_signals(
            signals_df, weights=blend_weights, method=blend_method, regime_probs=regime_probs
        )
        if composite.empty:
            return

        last_sig = composite.iloc[-1]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if last_sig > threshold:
            self.record_prediction(symbol, "UP", current_price)
            if position is None and current_price:
                qty = int(self.portfolio_value * 0.95 // current_price)
                if qty > 0:
                    order = self.create_order(symbol, qty, "buy")
                    self.submit_order(order)
        elif last_sig < -threshold:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)
