"""
Ensemble ML Strategy (RF + LightGBM)
--------------------------------------
Combines a scikit-learn Random Forest and LightGBM classifier via majority
voting.  A trade fires only when **both** classifiers agree on the direction.
Disagreement → NEUTRAL (no trade).  This is more conservative than either
model alone but produces higher-confidence signals.

Requires BOTH model files to exist for signals to fire:
  - models/classifier_rf.pkl   (sklearn)
  - models/classifier_lgb.txt  (LightGBM)

Train with:
    python train_ml_classifier.py
"""

from __future__ import annotations

import os
# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from lumibot.strategies.strategy import Strategy

from regime_engine.feature_extractor import get_regime_features
from regime_engine.ml_classifier import DirectionClassifier
from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier
from strategies.prediction_tracker import PredictionMixin


class EnsembleMLStrategy(PredictionMixin, Strategy):
    """
    Voting ensemble of Random Forest + LightGBM classifiers.

    Parameters
    ----------
    symbol           : str   — ticker to trade
    rf_model_path    : str   — path to the sklearn RF model .pkl
    lgb_model_path   : str   — path to the LightGBM model .txt
    lookback_days    : int   — history bars fed into feature extraction
    """

    parameters = {
        "symbol": "SPY",
        "rf_model_path": "models/classifier_rf.pkl",
        "lgb_model_path": "models/classifier_lgb.txt",
        "lookback_days": 130,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._init_predictions()

        rf_path  = self.parameters.get("rf_model_path",  "models/classifier_rf.pkl")
        lgb_path = self.parameters.get("lgb_model_path", "models/classifier_lgb.txt")

        self.rf_clf = DirectionClassifier(model_type="random_forest")
        self.rf_clf.load(rf_path)

        self.lgb_clf = LightGBMDirectionClassifier()
        self.lgb_clf.load(lgb_path)

    def on_trading_iteration(self) -> None:
        symbol: str   = self.parameters["symbol"]
        lookback: int = int(self.parameters.get("lookback_days", 130))

        bars = self.get_bars([symbol], lookback + 10, timestep="day")
        if not bars:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None or len(bars[asset_key].df) < 30:
            self.record_prediction(symbol, "NEUTRAL", self._safe_price(symbol))
            return

        df = bars[asset_key].df
        X  = get_regime_features(df, lookback=lookback)

        rf_pred  = self.rf_clf.predict(X)
        lgb_pred = self.lgb_clf.predict(X)

        # Only trade when both models agree
        if rf_pred != "NEUTRAL" and lgb_pred != "NEUTRAL" and rf_pred == lgb_pred:
            signal = rf_pred
        else:
            signal = "NEUTRAL"

        current_price = self._safe_price(symbol)
        self.record_prediction(symbol, signal, current_price)

        if signal == "UP":
            if self.get_position(symbol) is None:
                qty = int(self.portfolio_value * 0.95 // current_price)
                if qty > 0:
                    self.submit_order(self.create_order(symbol, qty, "buy"))
        elif signal == "DOWN":
            if self.get_position(symbol) is not None:
                self.sell_all()

    def _safe_price(self, symbol: str) -> float:
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception:
            return 0.0
