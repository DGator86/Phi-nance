"""
Calibration: MFM → direction probs and drift per horizon.

Heuristic V1: regime (trend/mean_revert) + sentiment (RSI, trend_alignment) + hedge (epp_proxy).
Replace with learned params from WF later.
"""

from __future__ import annotations

from phinence.contracts.projection_packet import DirectionProbs, Horizon
from phinence.mfm.merger import MarketFieldMap


def _clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, p))


def direction_from_mfm(mfm: MarketFieldMap, horizon: Horizon) -> DirectionProbs:
    """
    Map MFM to direction distribution. Heuristics:
    - Regime trend high → bias up; mean_revert high → bias flat.
    - RSI > 60 → slight down bias; RSI < 40 → slight up bias.
    - trend_alignment > 0 → up bias; < 0 → down bias.
    - epp_proxy (dealer) adds small tilt.
    """
    up, down, flat = 1.0 / 3, 1.0 / 3, 1.0 / 3
    regime = mfm.regime or {}
    sentiment = mfm.sentiment or {}
    hedge = mfm.hedge or {}
    rp = regime.get("regime_probs") or {}
    trend_p = float(rp.get("trend", 1.0 / 3))
    mean_revert_p = float(rp.get("mean_revert", 1.0 / 3))
    align = (sentiment.get("trend_alignment") or 0.0)
    rsi = float(sentiment.get("rsi") or 50)
    epp = float(hedge.get("epp_proxy") or 0)
    # Trend regime → directional bias
    up += trend_p * 0.15 + align * 0.2
    down -= trend_p * 0.1
    down += (rsi - 50) / 100
    up -= (rsi - 50) / 100
    flat += mean_revert_p * 0.1
    up += epp * 0.05
    down -= epp * 0.05
    up = _clamp_prob(up)
    down = _clamp_prob(down)
    flat = _clamp_prob(flat)
    total = up + down + flat
    if total <= 0:
        return DirectionProbs(up=1.0 / 3, down=1.0 / 3, flat=1.0 / 3)
    return DirectionProbs(up=up / total, down=down / total, flat=flat / total)


def drift_bps_from_mfm(mfm: MarketFieldMap, horizon: Horizon) -> float:
    """Drift in bps. Heuristic: regime trend and alignment."""
    regime = mfm.regime or {}
    sentiment = mfm.sentiment or {}
    rp = regime.get("regime_probs") or {}
    trend_p = float(rp.get("trend", 1.0 / 3))
    align = float(sentiment.get("trend_alignment") or 0)
    scale = 2.0 if horizon.value == "1d" else (0.5 if horizon.value == "5m" else 0.2)
    return (trend_p * align * 50 + align * 20) * scale
