"""
LightGBM Strategy
-----------------
Lumibot strategy backed by a pre-trained LightGBMDirectionClassifier.

If no model file exists at `models/classifier_lgb.txt`, the strategy
runs in observation-only mode (NEUTRAL), safe for dashboard backtesting.

Train with:
    python train_ml_classifier.py --model lightgbm
"""

from __future__ import annotations

import os
# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from lumibot.strategies.strategy import Strategy

from regime_engine.feature_extractor import get_regime_features
from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier
from strategies.prediction_tracker import PredictionMixin


class LightGBMStrategy(PredictionMixin, Strategy):
    """
    LightGBM gradient-boosting direction-prediction strategy.

    Parameters
    ----------
    symbol        : str   — ticker to trade (default 'SPY')
    model_path    : str   — path to the trained LightGBM model text file
    min_confidence: float — minimum P(direction) required to trade
    lookback_days : int   — bars of history to use for feature extraction
    """

    parameters = {
        "symbol": "SPY",
        "model_path": "models/classifier_lgb.txt",
        "min_confidence": 0.62,
        "lookback_days": 130,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._init_predictions()

        model_path = str(self.parameters.get("model_path", "models/classifier_lgb.txt"))
        self.classifier = LightGBMDirectionClassifier()
        self.classifier.load(model_path)

    def on_trading_iteration(self) -> None:
        symbol: str = str(self.parameters["symbol"])
        min_conf: float = float(str(self.parameters.get("min_confidence", 0.62)))
        lookback: int = int(str(self.parameters.get("lookback_days", 130)))

        bars = self.get_bars([symbol], lookback + 10, timestep="day")
        if not bars:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None or len(bars[asset_key].df) < 30:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        df = bars[asset_key].df
        try:
            X          = get_regime_features(df, lookback=lookback)
            direction  = self.classifier.predict(X)
            confidence = self.classifier.predict_proba(X)
        except Exception as exc:
            self.log_message(f"LightGBMStrategy: feature/predict error ({exc}); defaulting NEUTRAL")
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        current_price = self._safe_price(symbol)

        if direction == "NEUTRAL":
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        conf_score = confidence.get(direction, 0.5)

        if conf_score >= min_conf:
            if direction == "UP":
                self.record_prediction(symbol, "UP", current_price)
                if self.get_position(symbol) is None:
                    qty = int(self.portfolio_value * 0.95 // current_price)
                    if qty > 0:
                        self.submit_order(self.create_order(symbol, qty, "buy"))
            else:
                self.record_prediction(symbol, "DOWN", current_price)
                if self.get_position(symbol) is not None:
                    self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

    def _safe_price(self, symbol: str) -> float:
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception:
            return 0.0
