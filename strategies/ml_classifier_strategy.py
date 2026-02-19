"""
ML Classifier Strategy (scikit-learn)
--------------------------------------
Lumibot strategy that uses a pre-trained scikit-learn DirectionClassifier
to predict the next-day price direction and trade accordingly.

If no model file is found at `models/classifier_rf.pkl` the strategy runs
in observation-only mode (all NEUTRAL), making it safe to backtest without
pre-trained weights.

To train the model, run:
    python train_ml_classifier.py
"""

from __future__ import annotations

from lumibot.strategies.strategy import Strategy

from regime_engine.feature_extractor import get_regime_features
from regime_engine.ml_classifier import DirectionClassifier
from strategies.prediction_tracker import PredictionMixin


class MLClassifierStrategy(PredictionMixin, Strategy):
    """
    Scikit-learn Random Forest direction-prediction strategy.

    Parameters
    ----------
    symbol        : str   — ticker to trade (default 'SPY')
    model_type    : str   — 'random_forest' | 'gradient_boosting' | 'logistic'
    model_path    : str   — path to the trained .pkl file
    min_confidence: float — minimum P(direction) to place a trade [0.5, 1.0]
    lookback_days : int   — history bars to feed into feature extraction
    """

    parameters = {
        "symbol": "SPY",
        "model_type": "random_forest",
        "model_path": "models/classifier_rf.pkl",
        "min_confidence": 0.60,
        "lookback_days": 130,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._init_predictions()

        model_type = str(self.parameters.get("model_type", "random_forest"))
        model_path = str(self.parameters.get("model_path", "models/classifier_rf.pkl"))

        self.classifier = DirectionClassifier(model_type=model_type)
        self.classifier.load(model_path)

    def on_trading_iteration(self) -> None:
        symbol: str      = self.parameters["symbol"]
        min_conf: float  = float(self.parameters.get("min_confidence", 0.60))
        lookback: int    = int(self.parameters.get("lookback_days", 130))

        # ── 1. Gather recent price history ──────────────────────────
        bars = self.get_bars([symbol], lookback + 10, timestep="day")
        if not bars:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None or len(bars[asset_key].df) < 30:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        df = bars[asset_key].df

        # ── 2. Extract regime features ───────────────────────────────
        X = get_regime_features(df, lookback=lookback)

        # ── 3. Classify direction ────────────────────────────────────
        direction  = self.classifier.predict(X)
        confidence = self.classifier.predict_proba(X)

        current_price = self._safe_price(symbol)

        if direction == "NEUTRAL":
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        conf_score = confidence.get(direction, 0.5)

        # ── 4. Trade only when confidence exceeds threshold ──────────
        if conf_score >= min_conf:
            if direction == "UP":
                self.record_prediction(symbol, "UP", current_price)
                if self.get_position(symbol) is None:
                    qty = int(self.portfolio_value * 0.95 // current_price)
                    if qty > 0:
                        self.submit_order(self.create_order(symbol, qty, "buy"))
            else:  # DOWN
                self.record_prediction(symbol, "DOWN", current_price)
                if self.get_position(symbol) is not None:
                    self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _safe_price(self, symbol: str) -> float:
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception:
            return 0.0
