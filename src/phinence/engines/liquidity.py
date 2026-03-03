"""
Liquidity Engine V1 — bar-only. Volume profile (POC/VAH/VAL, HVN/LVN),
structural levels (swings, gaps, unfilled ranges, anchored VWAP).

Done when: levels match TradingView "close enough" across tickers/dates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phinence.contracts.assigned_packet import AssignedPacket


def volume_profile_poc_vah_val(df: pd.DataFrame, price_bins: int = 50) -> dict[str, float]:
    """POC, VAH, VAL from volume profile. df: OHLCV with close, volume."""
    if df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}
    low, high = df["low"].min(), df["high"].max()
    if low >= high:
        return {"poc": float(df["close"].iloc[-1]), "vah": high, "val": low}
    import numpy as np
    bins = np.linspace(low, high, price_bins + 1)
    df = df.copy()
    df["price_bin"] = np.digitize(df["close"], bins) - 1
    df["price_bin"] = df["price_bin"].clip(0, price_bins - 1)
    vol_by_bin = df.groupby("price_bin")["volume"].sum()
    if vol_by_bin.empty:
        return {"poc": float(df["close"].iloc[-1]), "vah": high, "val": low}
    poc_bin = vol_by_bin.idxmax()
    poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
    cum = vol_by_bin.sort_index().cumsum()
    total = cum.iloc[-1]
    if total <= 0:
        return {"poc": poc_price, "vah": high, "val": low}
    val_bin = (cum <= total * 0.16).idxmax() if (cum <= total * 0.16).any() else 0
    vah_bin = (cum >= total * 0.84).idxmax() if (cum >= total * 0.84).any() else len(bins) - 2
    val_price = (bins[val_bin] + bins[val_bin + 1]) / 2
    vah_price = (bins[vah_bin] + bins[vah_bin + 1]) / 2
    return {"poc": float(poc_price), "vah": float(vah_price), "val": float(val_price)}


def anchored_vwap(df: pd.DataFrame) -> float:
    """Anchored VWAP from session (full df)."""
    if df.empty or "volume" not in df.columns:
        return 0.0
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return float((typical * df["volume"]).sum() / df["volume"].replace(0, 1).sum())


def swing_high_low(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict[str, list[float]]:
    """Simple swing high/low: local extrema over left/right bars."""
    if df.empty or len(df) < left + right + 1:
        return {"swing_highs": [], "swing_lows": []}
    high = df["high"]
    low = df["low"]
    swing_highs = []
    swing_lows = []
    for i in range(left, len(df) - right):
        if high.iloc[i] == high.iloc[i - left : i + right + 1].max():
            swing_highs.append(float(high.iloc[i]))
        if low.iloc[i] == low.iloc[i - left : i + right + 1].min():
            swing_lows.append(float(low.iloc[i]))
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


class LiquidityEngine:
    """Bar-only liquidity: profile + structural levels."""

    def run(self, packet: AssignedPacket) -> dict[str, Any]:
        """Produce liquidity landmarks from 5m or 1m bars. No strategy/routing/sizing."""
        out: dict[str, Any] = {"poc": None, "vah": None, "val": None, "vwap": None, "swings": {}}
        if not packet.bars_5m and not packet.bars_1m:
            return out
        raw = packet.bars_5m if packet.bars_5m else packet.bars_1m
        df = pd.DataFrame(raw)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df.empty or len(df) < 2:
            return out
        profile = volume_profile_poc_vah_val(df)
        out["poc"] = profile["poc"]
        out["vah"] = profile["vah"]
        out["val"] = profile["val"]
        out["vwap"] = anchored_vwap(df)
        out["swings"] = swing_high_low(df)
        return out
