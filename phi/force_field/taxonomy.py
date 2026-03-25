"""Canonical sub-regime labels per family and simplex validation."""

from __future__ import annotations

import math
from typing import Mapping

INDICATOR_SUBREGIMES: tuple[str, ...] = (
    "trend_up",
    "trend_down",
    "range",
    "compression",
    "breakout_setup",
    "exhaustion",
    "reversal_setup",
)

LIQUIDITY_SUBREGIMES: tuple[str, ...] = (
    "thick_liquidity",
    "thin_liquidity",
    "pool_magnet",
    "vacuum_zone",
    "chop_friction",
    "sweep_risk",
)

GREEK_SUBREGIMES: tuple[str, ...] = (
    "pos_gamma_pin",
    "neg_gamma_accel",
    "vanna_dominant",
    "charm_dominant",
    "skew_stress",
    "vol_expansion",
    "vol_compression",
)

NEWS_SUBREGIMES: tuple[str, ...] = (
    "calm",
    "scheduled_event",
    "rumor_build",
    "live_shock",
    "post_event_decay",
    "macro_risk",
    "narrative_rotation",
)

INDUSTRY_SUBREGIMES: tuple[str, ...] = (
    "strong_index_lock",
    "strong_sector_lock",
    "local_idiosyncratic",
    "beta_expansion",
    "beta_compression",
    "sector_rotation",
    "divergence_reversion_setup",
)

FORCE_DIMS: tuple[str, ...] = (
    "directional",
    "magnet",
    "expansion",
    "friction",
    "systemic",
)


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    """Stable softmax over arbitrary keys."""
    if not logits:
        return {}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: v / s for k, v in exps.items()}


def uniform_simplex(keys: tuple[str, ...]) -> dict[str, float]:
    n = len(keys) or 1
    p = 1.0 / n
    return {k: p for k in keys}


def validate_simplex(probs: Mapping[str, float], keys: tuple[str, ...], tol: float = 1e-5) -> bool:
    """Return True if probs cover exactly ``keys`` and sum to 1 (within tol)."""
    got = set(probs.keys())
    if got != set(keys):
        return False
    return abs(sum(float(probs[k]) for k in keys) - 1.0) <= tol


def ensure_simplex(probs: Mapping[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    """Re-normalize to ``keys``; missing keys get 0 mass then uniform backfill if needed."""
    raw = {k: max(0.0, float(probs.get(k, 0.0))) for k in keys}
    s = sum(raw.values())
    if s <= 0.0:
        return uniform_simplex(keys)
    return {k: raw[k] / s for k in keys}
