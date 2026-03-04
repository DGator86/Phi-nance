"""
Sentiment Engine V1 — minimal. RSI + trend alignment + compression/expansion.

Done when: indicators match a reference implementation. Don't let it become a junk drawer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinence.contracts.assigned_packet import AssignedPacket


def rsi(close: pd.Series, period: int = 14) -> float:
    """RSI at last bar. Reference: Wilder."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def trend_alignment(close: pd.Series, fast: int = 9, slow: int = 21) -> float:
    """Signed strength: positive when fast > slow."""
    if len(close) < slow:
        return 0.0
    ema_f = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_s = close.ewm(span=slow, adjust=False).mean().iloc[-1]
    if ema_s == 0:
        return 0.0
    return float((ema_f - ema_s) / ema_s)


def compression_expansion(high: pd.Series, low: pd.Series, period: int = 10) -> str:
    """Range narrowing vs widening: 'compression' | 'expansion' | 'neutral'."""
    if len(high) < period * 2:
        return "neutral"
    recent_range = (high - low).tail(period).mean()
    prior_range = (high - low).tail(period * 2).head(period).mean()
    if prior_range <= 0:
        return "neutral"
    ratio = recent_range / prior_range
    if ratio < 0.8:
        return "compression"
    if ratio > 1.2:
        return "expansion"
    return "neutral"


class SentimentEngine:
    """Minimal: RSI + trend alignment + compression/expansion."""

    def run(self, packet: AssignedPacket) -> dict[str, Any]:
        """No strategy/routing/sizing."""
        out: dict[str, Any] = {"rsi": 50.0, "trend_alignment": 0.0, "range_state": "neutral"}
        if not packet.bars_5m and not packet.bars_1m:
            return out
        raw = packet.bars_5m if packet.bars_5m else packet.bars_1m
        df = pd.DataFrame(raw)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df.empty or len(df) < 15:
            return out
        out["rsi"] = rsi(df["close"])
        out["trend_alignment"] = trend_alignment(df["close"])
        out["range_state"] = compression_expansion(df["high"], df["low"])
        return out
