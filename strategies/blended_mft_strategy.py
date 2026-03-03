"""
BlendedMFTStrategy
------------------
A Lumibot strategy that runs the full MFT RegimeEngine pipeline and lets
the user tune per-indicator blend weights at runtime.

The composite signal is re-weighted by the user's indicator_weights dict
before the confidence gate is applied. The signal sign drives direction:
  > +threshold → BUY
  < -threshold → SELL
  otherwise   → hold / close

Parameters (passed via dashboard or run_backtest):
    symbol            : str   — ticker to trade (default "SPY")
    indicator_weights : dict  — {indicator_name: weight_multiplier}
                                defaults to all 1.0 (standard MFT)
    signal_threshold  : float — |composite| must exceed this to trade (default 0.15)
    confidence_floor  : float — minimum c_field * c_consensus to trade (default 0.3)
    lookback_bars     : int   — how many bars of OHLCV history to pass to the engine
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

import numpy as np
import pandas as pd
import yaml

from lumibot.strategies import Strategy

from strategies.prediction_tracker import PredictionMixin

# ---------------------------------------------------------------------------
# Config cache (loaded once per process)
# ---------------------------------------------------------------------------
_CACHED_CFG: Optional[dict] = None


def _get_config() -> dict:
    global _CACHED_CFG
    if _CACHED_CFG is None:
        cfg_path = os.path.join("regime_engine", "config.yaml")
        with open(cfg_path) as f:
            _CACHED_CFG = yaml.safe_load(f)
    return _CACHED_CFG


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class BlendedMFTStrategy(PredictionMixin, Strategy):
    """
    Full MFT strategy with user-tunable indicator blend weights.

    All indicator weights default to 1.0 (standard MFT configuration).
    Increasing a weight boosts that indicator's contribution to the
    composite signal; setting it to 0.0 mutes it entirely.
    """

    parameters = {
        "symbol":            "SPY",
        "indicator_weights": {},       # {indicator_name: multiplier}; {} = all 1.0
        "signal_threshold":  0.15,
        "confidence_floor":  0.30,
        "lookback_bars":     500,
    }

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from regime_engine.scanner import RegimeEngine
            self._engine = RegimeEngine(_get_config())
        return self._engine

    # ── Main loop ──────────────────────────────────────────────────────────
    def on_trading_iteration(self):
        symbol     = str(self.parameters.get("symbol", "SPY"))
        weights    = dict(self.parameters.get("indicator_weights", {}))
        threshold  = float(self.parameters.get("signal_threshold", 0.15))
        conf_floor = float(self.parameters.get("confidence_floor", 0.30))
        lookback   = int(self.parameters.get("lookback_bars", 500))

        # ── Fetch OHLCV ───────────────────────────────────────────────────
        bars = self.get_historical_prices(symbol, lookback, "day")
        if bars is None or bars.df is None or len(bars.df) < 120:
            return

        ohlcv = bars.df.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(ohlcv.columns):
            return

        # ── Run MFT engine ────────────────────────────────────────────────
        try:
            engine = self._get_engine()
            out    = engine.run(ohlcv)
        except Exception:
            return

        mix_df    = out["mix"]
        signals   = out["signals"]
        wt_df     = out["weights"]

        if mix_df.empty or signals.empty:
            return

        # ── Apply custom indicator weights ────────────────────────────────
        # Re-compute a weighted composite signal from scratch
        if weights:
            adj_signals  = signals.copy()
            adj_weights  = wt_df.copy()
            for ind, mult in weights.items():
                if ind in adj_signals.columns:
                    adj_signals[ind] = adj_signals[ind] * float(mult)
                if ind in adj_weights.columns:
                    adj_weights[ind] = adj_weights[ind] * float(mult)

            # Weighted mean of adjusted signals (last bar)
            w_last = adj_weights.iloc[-1].clip(lower=0)
            s_last = adj_signals.iloc[-1]
            denom  = w_last.sum() + 1e-9
            composite = float((w_last * s_last).sum() / denom)
        else:
            composite = float(mix_df["composite_signal"].iloc[-1])

        # ── Confidence gate ───────────────────────────────────────────────
        c_field     = float(mix_df["c_field"].iloc[-1])
        c_consensus = float(mix_df["c_consensus"].iloc[-1])
        confidence  = c_field * c_consensus

        # ── Current price + signal ────────────────────────────────────────
        price = self.get_last_price(symbol)
        if price is None:
            return

        # Map composite to direction for prediction tracker
        if confidence >= conf_floor and abs(composite) >= threshold:
            signal = "UP" if composite > 0 else "DOWN"
        else:
            signal = "NEUTRAL"

        self.record_prediction(symbol, signal, price)

        # ── Position management ───────────────────────────────────────────
        position = self.get_position(symbol)
        cash     = self.get_cash()

        if signal == "UP":
            if position is None or position.quantity <= 0:
                # Buy with available cash
                qty = int(cash * 0.95 / price)
                if qty > 0:
                    order = self.create_order(symbol, qty, "buy")
                    self.submit_order(order)

        elif signal == "DOWN":
            if position is not None and position.quantity > 0:
                self.sell_all()

        # NEUTRAL: hold existing

    def on_finish(self):
        pass
