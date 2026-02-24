"""
PlutusStrategy
--------------
A Lumibot strategy powered by the 0xroyce/plutus LLM via Ollama.

The strategy follows this pipeline on every trading bar:

  1. Fetch recent OHLCV history from the broker/data source.
  2. Run the MFT RegimeEngine to produce quantitative signals
     (regime probabilities, composite signal, confidence scores).
  3. Build a structured market brief (OHLCV summary + MFT context).
  4. Send the brief to the Plutus LLM and receive a JSON decision:
       {"signal": "BUY"|"SELL"|"HOLD", "confidence": 0-1,
        "reasoning": "...", "risk_note": "..."}
  5. Gate the decision against a minimum confidence threshold.
  6. Execute orders; record the prediction for the accuracy tracker.
  7. On position close, feed the exit price back into the advisor's
     trade journal so Plutus can learn from past outcomes.

Parameters
----------
symbol          : str   — ticker to trade              (default "SPY")
min_confidence  : float — min LLM confidence to trade  (default 0.60)
lookback_bars   : int   — OHLCV bars to feed the engine (default 300)
ollama_host     : str   — Ollama server URL            (default "http://localhost:11434")
plutus_model    : str   — Ollama model tag             (default "0xroyce/plutus")
position_pct    : float — fraction of cash to deploy   (default 0.95)

Prerequisites
-------------
  1. `ollama` installed and running locally (or remote host configured)
  2. Model pulled:  ollama pull 0xroyce/plutus
"""

from __future__ import annotations

import os

# Force backtesting mode before any lumibot import
os.environ.setdefault("IS_BACKTESTING", "True")

import logging
from typing import Optional

import yaml

from lumibot.strategies import Strategy
from strategies.prediction_tracker import PredictionMixin

logger = logging.getLogger(__name__)

_CACHED_CFG: Optional[dict] = None


def _get_config() -> dict:
    global _CACHED_CFG
    if _CACHED_CFG is None:
        path = os.path.join("regime_engine", "config.yaml")
        with open(path) as f:
            _CACHED_CFG = yaml.safe_load(f)
    return _CACHED_CFG


class PlutusStrategy(PredictionMixin, Strategy):
    """
    LLM-guided trading strategy using the Plutus finance model via Ollama.

    The model receives a structured market brief each bar and returns a
    BUY / SELL / HOLD decision with confidence and reasoning.  A rolling
    trade journal is maintained so the model can observe past outcomes and
    adapt its recommendations (in-context learning).
    """

    parameters = {
        "symbol":         "SPY",
        "min_confidence": 0.60,
        "lookback_bars":  300,
        "ollama_host":    "http://localhost:11434",
        "plutus_model":   "0xroyce/plutus",
        "position_pct":   0.95,
    }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._init_predictions()
        self._engine  = None
        self._advisor = None
        self._prev_signal = "HOLD"
        self._prev_price: Optional[float] = None
        self._setup_advisor()

    def _setup_advisor(self) -> None:
        from regime_engine.plutus_advisor import PlutusAdvisor
        host  = str(self.parameters.get("ollama_host", "http://localhost:11434"))
        model = str(self.parameters.get("plutus_model", "0xroyce/plutus"))
        min_c = float(self.parameters.get("min_confidence", 0.60))
        self._advisor = PlutusAdvisor(model=model, host=host, min_conf=min_c)

    def _get_engine(self):
        if self._engine is None:
            from regime_engine.scanner import RegimeEngine
            self._engine = RegimeEngine(_get_config())
        return self._engine

    # ── Main loop ─────────────────────────────────────────────────────────────

    def on_trading_iteration(self) -> None:
        symbol       = str(self.parameters.get("symbol",         "SPY"))
        min_conf     = float(self.parameters.get("min_confidence", 0.60))
        lookback     = int(self.parameters.get("lookback_bars",   300))
        position_pct = float(self.parameters.get("position_pct",  0.95))

        # ── 1. Current price ─────────────────────────────────────────────
        price = self._safe_price(symbol)
        if not price:
            return

        # ── 2. Record outcome of the previous bar's open position ────────
        if self._prev_signal == "BUY" and self._prev_price:
            if self._advisor is not None:
                self._advisor.record_outcome(symbol, price)
        self._prev_price = price

        # ── 3. Fetch OHLCV ───────────────────────────────────────────────
        bars = self.get_historical_prices(symbol, lookback, "day")
        if bars is None or bars.df is None or len(bars.df) < 60:
            self.record_prediction(symbol, "NEUTRAL", price)
            return

        ohlcv = bars.df.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        if not {"open", "high", "low", "close", "volume"}.issubset(ohlcv.columns):
            self.record_prediction(symbol, "NEUTRAL", price)
            return

        # ── 4. Run MFT engine for quantitative context ───────────────────
        mft_out = None
        try:
            mft_out = self._get_engine().run(ohlcv)
        except Exception as e:
            logger.warning("MFT engine error (non-fatal): %s", e)

        # ── 5. Ask Plutus ────────────────────────────────────────────────
        if self._advisor is None:
            self.record_prediction(symbol, "NEUTRAL", price)
            return

        # Check availability gracefully — fall through to NEUTRAL if offline
        if not self._advisor.is_available():
            logger.warning("Plutus/Ollama not available — holding position")
            self.record_prediction(symbol, "NEUTRAL", price)
            return

        from regime_engine.plutus_advisor import build_market_brief
        ohlcv_summary, mft_signals = build_market_brief(ohlcv, mft_out)

        try:
            decision = self._advisor.recommend(symbol, ohlcv_summary, mft_signals, price)
        except Exception as e:
            logger.error("Plutus recommendation error: %s", e)
            self.record_prediction(symbol, "NEUTRAL", price)
            return

        # ── 6. Map to tracker signal ─────────────────────────────────────
        if decision.signal == "BUY":
            signal = "UP"
        elif decision.signal == "SELL":
            signal = "DOWN"
        else:
            signal = "NEUTRAL"

        self.record_prediction(symbol, signal, price)
        self._prev_signal = decision.signal

        # ── 7. Execute trades ────────────────────────────────────────────
        position = self.get_position(symbol)
        cash     = self.get_cash()

        if decision.is_actionable(min_conf):
            if decision.signal == "BUY":
                if position is None or position.quantity <= 0:
                    qty = int(cash * position_pct / price)
                    if qty > 0:
                        logger.info(
                            "Plutus BUY %s x%d @ %.2f — %s (conf=%.2f)",
                            symbol, qty, price, decision.reasoning, decision.confidence,
                        )
                        self.submit_order(self.create_order(symbol, qty, "buy"))

            elif decision.signal == "SELL":
                if position is not None and position.quantity > 0:
                    logger.info(
                        "Plutus SELL %s @ %.2f — %s (conf=%.2f)",
                        symbol, price, decision.reasoning, decision.confidence,
                    )
                    self.sell_all()

        # else: HOLD or below threshold — maintain current position

    def on_finish(self) -> None:
        pass

    # ── Helper ────────────────────────────────────────────────────────────────

    def _safe_price(self, symbol: str) -> float:
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception:
            return 0.0
