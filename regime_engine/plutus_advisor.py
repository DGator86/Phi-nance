"""
PlutusAdvisor
-------------
Interfaces with the Ollama-hosted 0xroyce/plutus model (a finance-
specialized LLaMA 3.1-8B) to generate trading recommendations.

The advisor maintains an in-context learning journal — a rolling log of
past decisions and their realised outcomes.  On every call the journal is
injected into the conversation so Plutus can see what worked and adapt.

Ollama REST API endpoint (local): http://localhost:11434/api/chat
Pull model: ollama pull 0xroyce/plutus
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL   = "0xroyce/plutus"
_DEFAULT_HOST    = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120          # seconds — LLM inference can be slow locally
_JOURNAL_SIZE    = 20           # how many past decisions to keep in context
_SYSTEM_PROMPT   = """\
You are Plutus, an expert quantitative trading assistant trained on financial \
literature, technical analysis, options theory, risk management, and \
behavioural finance.

You receive a structured market brief each bar and must return a JSON decision \
object. Follow these rules strictly:

1. Analyse the provided OHLCV summary, regime probabilities, and composite \
signal from the Market Field Theory (MFT) engine.
2. Review the trade journal — a history of past decisions and their outcomes. \
Adapt your strategy based on what has worked and what has not.
3. Return ONLY valid JSON in the following schema — no other text:
   {
     "signal": "BUY" | "SELL" | "HOLD",
     "confidence": <float 0-1>,
     "reasoning": "<one concise sentence>",
     "risk_note": "<one sentence on key risk>"
   }
4. Use high confidence (> 0.7) only when multiple regime signals agree.
5. Prefer HOLD when uncertain; capital preservation matters.
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class JournalEntry:
    """A single past decision with its observed outcome."""
    date:       str
    symbol:     str
    signal:     str          # BUY | SELL | HOLD
    confidence: float
    entry_price: float
    exit_price:  Optional[float] = None   # None while still open
    pnl_pct:     Optional[float] = None   # realised P&L %
    correct:     Optional[bool]  = None   # was the direction right?


@dataclass
class PlutusDecision:
    """Structured response from the Plutus model."""
    signal:     str           # BUY | SELL | HOLD
    confidence: float         # 0-1
    reasoning:  str
    risk_note:  str
    raw:        str = ""      # raw LLM text (for debugging)

    def is_actionable(self, min_conf: float = 0.55) -> bool:
        return self.signal in ("BUY", "SELL") and self.confidence >= min_conf


# ---------------------------------------------------------------------------
# PlutusAdvisor
# ---------------------------------------------------------------------------
class PlutusAdvisor:
    """
    Manages the Plutus LLM, its trade journal, and trade recommendation logic.

    Parameters
    ----------
    model       : Ollama model tag, default "0xroyce/plutus"
    host        : Ollama base URL, default "http://localhost:11434"
    journal_size: max past decisions kept in context (in-context learning)
    min_conf    : minimum confidence required for BUY/SELL signals
    """

    def __init__(
        self,
        model:        str   = _DEFAULT_MODEL,
        host:         str   = _DEFAULT_HOST,
        journal_size: int   = _JOURNAL_SIZE,
        min_conf:     float = 0.55,
        timeout:      int   = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model    = os.getenv("PLUTUS_MODEL", model)
        self.host     = os.getenv("OLLAMA_HOST", host).rstrip("/")
        self.min_conf = min_conf
        self.timeout  = timeout
        self._journal: Deque[JournalEntry] = deque(maxlen=journal_size)
        self._chat_url = f"{self.host}/api/chat"

    # ── Health check ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and the model is loaded."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            tags = r.json().get("models", [])
            return any(self.model in (t.get("name", "") or t.get("model", ""))
                       for t in tags)
        except Exception:
            return False

    def pull_model(self) -> bool:
        """Ask Ollama to pull the model if not already present. Returns success."""
        try:
            r = requests.post(
                f"{self.host}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=600,
            )
            return r.status_code == 200
        except Exception as e:
            logger.error("Plutus pull failed: %s", e)
            return False

    # ── Core recommendation ───────────────────────────────────────────────────

    def recommend(
        self,
        symbol:          str,
        ohlcv_summary:   Dict[str, Any],
        mft_signals:     Dict[str, Any],
        current_price:   float,
    ) -> PlutusDecision:
        """
        Ask Plutus for a trading decision given market context.

        Parameters
        ----------
        symbol        : ticker being evaluated
        ohlcv_summary : dict with keys like close, volume, rsi, atr, returns
        mft_signals   : dict from the MFT engine (regime probs, composite, etc.)
        current_price : latest price (for journal entry)

        Returns
        -------
        PlutusDecision
        """
        user_msg = self._build_user_message(symbol, ohlcv_summary, mft_signals)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        raw = self._call_ollama(messages)
        decision = self._parse_decision(raw)

        # Record open position in journal (outcome filled later)
        self._journal.append(JournalEntry(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            symbol=symbol,
            signal=decision.signal,
            confidence=decision.confidence,
            entry_price=current_price,
        ))

        return decision

    # ── Journal management ────────────────────────────────────────────────────

    def record_outcome(
        self,
        symbol:     str,
        exit_price: float,
    ) -> None:
        """
        Fill in the exit price and P&L for the most recent open position
        for this symbol in the journal.
        """
        for entry in reversed(self._journal):
            if entry.symbol == symbol and entry.exit_price is None:
                entry.exit_price = exit_price
                if entry.entry_price and entry.entry_price > 0:
                    pct = (exit_price - entry.entry_price) / entry.entry_price
                    entry.pnl_pct = round(pct * 100, 3)
                    if entry.signal == "BUY":
                        entry.correct = pct > 0
                    elif entry.signal == "SELL":
                        entry.correct = pct < 0
                    else:
                        entry.correct = None
                break

    def get_journal(self) -> List[Dict]:
        return [asdict(e) for e in self._journal]

    def journal_accuracy(self) -> float:
        decided = [e for e in self._journal if e.correct is not None]
        if not decided:
            return 0.0
        return sum(1 for e in decided if e.correct) / len(decided)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        symbol:        str,
        ohlcv_summary: Dict[str, Any],
        mft_signals:   Dict[str, Any],
    ) -> str:
        lines: List[str] = [
            f"=== MARKET BRIEF — {symbol} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===",
            "",
            "--- OHLCV Summary (last bar + recent stats) ---",
        ]
        for k, v in ohlcv_summary.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        lines += ["", "--- MFT Regime Engine Signals ---"]
        for k, v in mft_signals.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        # Inject trade journal for in-context learning
        if self._journal:
            lines += ["", "--- Trade Journal (past decisions & outcomes) ---"]
            for e in self._journal:
                outcome = ""
                if e.pnl_pct is not None:
                    tag = "CORRECT" if e.correct else "WRONG"
                    outcome = f" | exit={e.exit_price:.2f} | pnl={e.pnl_pct:+.2f}% | {tag}"
                lines.append(
                    f"  [{e.date}] {e.symbol} {e.signal} conf={e.confidence:.2f}"
                    f" entry={e.entry_price:.2f}{outcome}"
                )
            hit_rate = self.journal_accuracy()
            if hit_rate:
                lines.append(f"  Journal hit-rate: {hit_rate:.1%}")

        lines += [
            "",
            "Based on the above, provide your trading decision as JSON.",
        ]
        return "\n".join(lines)

    def _call_ollama(self, messages: List[Dict]) -> str:
        try:
            payload = {
                "model":    self.model,
                "messages": messages,
                "stream":   False,
                "options": {
                    "temperature": 0.4,  # lower temp → more consistent decisions
                    "top_p":       0.9,
                    "top_k":       40,
                },
            }
            r = requests.post(self._chat_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            return ""

    @staticmethod
    def _parse_decision(raw: str) -> PlutusDecision:
        """
        Extract JSON from the raw LLM output.  Falls back to HOLD on any
        parsing failure so the strategy degrades gracefully.
        """
        fallback = PlutusDecision(
            signal="HOLD", confidence=0.0,
            reasoning="Parse error — defaulting to HOLD",
            risk_note="", raw=raw,
        )
        if not raw:
            return fallback

        # Try to extract a JSON block from the response
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not json_match:
            return fallback

        try:
            obj = json.loads(json_match.group())
        except json.JSONDecodeError:
            return fallback

        signal = str(obj.get("signal", "HOLD")).upper().strip()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"

        try:
            conf = float(obj.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.5

        return PlutusDecision(
            signal=signal,
            confidence=conf,
            reasoning=str(obj.get("reasoning", ""))[:300],
            risk_note=str(obj.get("risk_note", ""))[:300],
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_market_brief(ohlcv_df, mft_out: Optional[Dict] = None) -> tuple[Dict, Dict]:
    """
    Convert a raw OHLCV DataFrame + optional MFT engine output into the
    two dicts expected by PlutusAdvisor.recommend().

    Returns (ohlcv_summary, mft_signals).
    """
    import numpy as np

    df = ohlcv_df.copy()
    df.columns = [c.lower() for c in df.columns]
    close  = df["close"]
    volume = df.get("volume", None)

    ret_1d  = float(close.pct_change().iloc[-1]) if len(close) > 1 else 0.0
    ret_5d  = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    vol_20  = float(close.pct_change().rolling(20).std().iloc[-1]) if len(close) >= 20 else 0.0

    ohlcv_summary: Dict[str, Any] = {
        "close":          round(float(close.iloc[-1]), 4),
        "open":           round(float(df["open"].iloc[-1]), 4)   if "open"   in df else None,
        "high":           round(float(df["high"].iloc[-1]), 4)   if "high"   in df else None,
        "low":            round(float(df["low"].iloc[-1]), 4)    if "low"    in df else None,
        "return_1d_pct":  round(ret_1d  * 100, 3),
        "return_5d_pct":  round(ret_5d  * 100, 3),
        "return_20d_pct": round(ret_20d * 100, 3),
        "vol_20d_pct":    round(vol_20  * 100, 3),
        "bars_available": len(df),
    }
    if volume is not None:
        vol_avg  = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        vol_last = float(volume.iloc[-1])
        ohlcv_summary["volume_ratio_vs_avg"] = round(vol_last / (vol_avg + 1e-9), 3)

    # MFT signals
    mft_signals: Dict[str, Any] = {}
    if mft_out is not None:
        try:
            mix = mft_out.get("mix", None)
            if mix is not None and not mix.empty:
                last = mix.iloc[-1]
                for col in ("composite_signal", "score", "c_field", "c_consensus", "c_liquidity"):
                    if col in last.index:
                        mft_signals[col] = round(float(last[col]), 4)

            regime_probs = mft_out.get("regime_probs", None)
            if regime_probs is not None and not regime_probs.empty:
                top = regime_probs.iloc[-1].sort_values(ascending=False)
                mft_signals["top_regime"]       = str(top.index[0])
                mft_signals["top_regime_prob"]  = round(float(top.iloc[0]), 4)
                mft_signals["second_regime"]    = str(top.index[1]) if len(top) > 1 else ""
                mft_signals["second_regime_prob"] = round(float(top.iloc[1]), 4) if len(top) > 1 else 0.0
        except Exception as e:
            mft_signals["mft_error"] = str(e)
    else:
        mft_signals["note"] = "MFT engine output not provided"

    return ohlcv_summary, mft_signals
