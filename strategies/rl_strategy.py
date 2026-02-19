"""
RL Strategy (Stable-Baselines3 PPO)
--------------------------------------
Lumibot strategy backed by a pre-trained Stable-Baselines3 PPO agent.

The agent maps regime-feature observations to one of four actions:
  0 = HOLD  →  no trade recorded
  1 = BUY   →  record UP, enter long
  2 = SELL  →  record DOWN, exit long
  3 = EXIT  →  record DOWN, exit long

If no model file exists at `models/rl_agent_ppo.zip`, the strategy runs
in NEUTRAL mode, safe for backtesting without pre-trained weights.

Requirements:
    pip install stable-baselines3 gymnasium

Train with:
    python train_rl_agent.py
"""

from __future__ import annotations

import numpy as np
from lumibot.strategies.strategy import Strategy

from regime_engine.feature_extractor import get_regime_features
from regime_engine.features import FeatureEngine
from strategies.prediction_tracker import PredictionMixin

try:
    from stable_baselines3 import PPO as _PPO
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class RLStrategy(PredictionMixin, Strategy):
    """
    Stable-Baselines3 PPO direction-prediction strategy.

    Parameters
    ----------
    symbol        : str  — ticker to trade
    model_path    : str  — path to the PPO model (.zip)
    lookback_days : int  — history bars for feature extraction
    """

    parameters = {
        "symbol": "SPY",
        "model_path": "models/rl_agent_ppo.zip",
        "lookback_days": 130,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._init_predictions()
        self.rl_model = None

        if not _SB3_AVAILABLE:
            print(
                "[RLStrategy] stable-baselines3 not installed — "
                "running NEUTRAL. Install with: pip install stable-baselines3 gymnasium"
            )
            return

        import os
        model_path = self.parameters.get("model_path", "models/rl_agent_ppo.zip")
        if not os.path.exists(model_path):
            print(
                f"[RLStrategy] No model at {model_path} — running NEUTRAL. "
                "Train with: python train_rl_agent.py"
            )
            return

        try:
            self.rl_model = _PPO.load(model_path)
            print(f"[RLStrategy] Loaded PPO agent from {model_path}")
        except Exception as exc:
            print(f"[RLStrategy] Failed to load model: {exc} — running NEUTRAL.")

    def on_trading_iteration(self) -> None:
        symbol: str   = self.parameters["symbol"]
        lookback: int = int(self.parameters.get("lookback_days", 130))
        current_price = self._safe_price(symbol)

        if self.rl_model is None:
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        bars = self.get_bars([symbol], lookback + 10, timestep="day")
        if not bars:
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None or len(bars[asset_key].df) < 30:
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        df = bars[asset_key].df
        X  = get_regime_features(df, lookback=lookback)

        # Build observation matching TradingEnv._get_obs()
        feature_vals = X.values.astype(np.float32).flatten()
        extra = np.array([
            1.0,   # cash ratio (we don't track full env state here)
            0.0,   # position ratio placeholder
            1.0,   # normalised price level placeholder
        ], dtype=np.float32)
        obs = np.concatenate([feature_vals, extra])

        action, _ = self.rl_model.predict(obs, deterministic=True)
        action = int(action)

        if action == 1:   # BUY
            self.record_prediction(symbol, "UP", current_price)
            if self.get_position(symbol) is None:
                qty = int(self.portfolio_value * 0.95 // (current_price or 1))
                if qty > 0:
                    self.submit_order(self.create_order(symbol, qty, "buy"))
        elif action in (2, 3):   # SELL / EXIT
            self.record_prediction(symbol, "DOWN", current_price)
            if self.get_position(symbol) is not None:
                self.sell_all()
        else:  # HOLD
            self.record_prediction(symbol, "NEUTRAL", current_price)

    def _safe_price(self, symbol: str) -> float:
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception:
            return 0.0
