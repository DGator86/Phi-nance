"""
Regime Engine V1 — deterministic. ER + ATR% + EMA alignment → soft regime probabilities.

Done when: random sampled windows "look right"; confidence behaves sensibly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinence.contracts.assigned_packet import AssignedPacket


def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as % of close (annualized proxy)."""
    if df.empty or len(df) < period + 1:
        return 0.0
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float((atr / close.iloc[-1]) * 100) if close.iloc[-1] else 0.0


def ema_alignment(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> str:
    """Bullish / bearish / neutral from EMA cross."""
    if df.empty or len(df) < slow:
        return "neutral"
    close = df["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_s = close.ewm(span=slow, adjust=False).mean().iloc[-1]
    if ema_f > ema_s * 1.001:
        return "bullish"
    if ema_f < ema_s * 0.999:
        return "bearish"
    return "neutral"


def regime_probs_from_er_atr_ema(
    df: pd.DataFrame,
    er_period: int = 10,
    atr_period: int = 14,
) -> dict[str, float]:
    """
    Soft regime: trend / mean_revert / expansion from ER + ATR% + EMA.
    Deterministic V1; no HMM.
    """
    if df.empty or len(df) < max(er_period, atr_period) + 5:
        return {"trend": 1.0 / 3, "mean_revert": 1.0 / 3, "expansion": 1.0 / 3}
    close = df["close"]
    # ER (Efficiency Ratio) — trend strength
    change = abs(close.iloc[-1] - close.iloc[-er_period])
    volatility = close.diff().abs().tail(er_period).sum()
    er = float(change / volatility) if volatility else 0.0
    atr_p = atr_pct(df, atr_period)
    align = ema_alignment(df)
    # Map to soft probs
    trend_p = min(1.0, er * 0.5 + (0.4 if align != "neutral" else 0.0))
    expansion_p = min(1.0, atr_p / 2.0) if atr_p > 0 else 0.0
    mean_revert_p = 1.0 - trend_p - expansion_p
    mean_revert_p = max(0.0, min(1.0, mean_revert_p))
    total = trend_p + expansion_p + mean_revert_p
    if total <= 0:
        return {"trend": 1.0 / 3, "mean_revert": 1.0 / 3, "expansion": 1.0 / 3}
    return {
        "trend": trend_p / total,
        "mean_revert": mean_revert_p / total,
        "expansion": expansion_p / total,
    }


class RegimeEngine:
    """Deterministic regime: ER + ATR% + EMA → soft probabilities."""

    def run(self, packet: AssignedPacket) -> dict[str, Any]:
        """No strategy/routing/sizing. Confidence from bar coverage."""
        out: dict[str, Any] = {"regime_probs": {}, "alignment": "neutral", "atr_pct": 0.0}
        if not packet.bars_5m and not packet.bars_1m:
            return out
        raw = packet.bars_5m if packet.bars_5m else packet.bars_1m
        df = pd.DataFrame(raw)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df.empty or len(df) < 20:
            return out
        out["regime_probs"] = regime_probs_from_er_atr_ema(df)
        out["alignment"] = ema_alignment(df)
        out["atr_pct"] = atr_pct(df)
        return out
