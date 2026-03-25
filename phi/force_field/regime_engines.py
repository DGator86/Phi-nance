"""Rule-based regime probability builders from a flat numeric feature dict.

These are **starting priors**: wire richer features from ``regime_engine`` / vendors
without changing the force-field or strategy layers.
"""

from __future__ import annotations

from typing import Mapping

from phi.force_field.schemas import RegimeTensor
from phi.force_field.taxonomy import (
    GREEK_SUBREGIMES,
    INDICATOR_SUBREGIMES,
    INDUSTRY_SUBREGIMES,
    LIQUIDITY_SUBREGIMES,
    NEWS_SUBREGIMES,
    ensure_simplex,
    softmax,
    uniform_simplex,
)


def _f(features: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return float(features.get(key, default))


def indicator_from_features(features: Mapping[str, float]) -> dict[str, float]:
    """Heuristic logits from common indicator names (all optional)."""
    adx = _f(features, "adx", 20.0)
    rsi = _f(features, "rsi", 50.0)
    bb_width_pct = _f(features, "bb_width_pct", 0.04)
    vwap_z = _f(features, "vwap_z", 0.0)
    donchian_dist = _f(features, "donchian_dist_pct", 0.0)
    eff_ratio = _f(features, "kaufman_efficiency", 0.5)

    logits = {
        "trend_up": (adx - 22.0) + 2.0 * max(0.0, vwap_z) + 3.0 * (eff_ratio - 0.45),
        "trend_down": (adx - 22.0) + 2.0 * max(0.0, -vwap_z) + 3.0 * (eff_ratio - 0.45),
        "range": (25.0 - adx) - 2.0 * abs(vwap_z),
        "compression": max(0.0, 0.035 - bb_width_pct) * 80.0,
        "breakout_setup": abs(donchian_dist) * 25.0 + max(0.0, 0.04 - bb_width_pct) * 40.0,
        "exhaustion": max(0.0, rsi - 72.0) / 4.0 + max(0.0, 28.0 - rsi) / 4.0,
        "reversal_setup": max(0.0, 70.0 - rsi) / 5.0 + max(0.0, rsi - 30.0) / 5.0 + (22.0 - adx) * 0.2,
    }
    return ensure_simplex(softmax(logits), INDICATOR_SUBREGIMES)


def liquidity_from_features(features: Mapping[str, float]) -> dict[str, float]:
    spread_pct = _f(features, "spread_pct", 0.0005)
    imb = _f(features, "book_imbalance", 0.0)
    vol_pct = _f(features, "volume_percentile", 0.5)
    pool_prox = _f(features, "liquidity_pool_proximity", 0.0)
    void_score = _f(features, "book_void_score", 0.0)

    thick = 1.0 if spread_pct <= 0.002 and vol_pct >= 0.4 else 0.0
    thin = 1.0 - thick
    logits = {
        "thick_liquidity": 2.2 * thick + 0.4 * vol_pct,
        "thin_liquidity": 2.0 * thin + 3.0 * spread_pct,
        "pool_magnet": 1.5 * max(0.0, pool_prox),
        "vacuum_zone": 2.0 * void_score + 1.0 * thin,
        "chop_friction": 0.8 * thick + 0.6 * abs(imb),
        "sweep_risk": 1.5 * thin + 0.5 * abs(imb) + 2.0 * max(0.0, spread_pct - 0.003),
    }
    return ensure_simplex(softmax(logits), LIQUIDITY_SUBREGIMES)


def greek_from_features(features: Mapping[str, float]) -> dict[str, float]:
    iv_rank = _f(features, "iv_rank", 0.5)
    gamma_tilt = _f(features, "net_gex_proxy", 0.0)
    vanna_z = _f(features, "vanna_z", 0.0)
    charm_z = _f(features, "charm_z", 0.0)
    skew_pct = _f(features, "skew_percentile", 0.5)
    fb_iv_spread = _f(features, "front_back_iv_spread", 0.0)

    logits = {
        "pos_gamma_pin": max(0.0, gamma_tilt) * 3.0 + 0.5 * (1.0 - iv_rank),
        "neg_gamma_accel": max(0.0, -gamma_tilt) * 3.0 + 0.5 * iv_rank,
        "vanna_dominant": abs(vanna_z) * 1.2,
        "charm_dominant": abs(charm_z) * 1.2,
        "skew_stress": abs(skew_pct - 0.5) * 4.0,
        "vol_expansion": (iv_rank - 0.55) * 4.0 + max(0.0, fb_iv_spread) * 2.0,
        "vol_compression": (0.45 - iv_rank) * 4.0 + max(0.0, -fb_iv_spread) * 2.0,
    }
    return ensure_simplex(softmax(logits), GREEK_SUBREGIMES)


def news_from_features(features: Mapping[str, float]) -> dict[str, float]:
    news_rate = _f(features, "headline_velocity", 0.0)
    shock = _f(features, "sentiment_impulse", 0.0)
    sched = _f(features, "scheduled_event_hours", 1e6)
    decay = _f(features, "post_event_decay_clock", 0.0)

    near_event = 1.0 if sched <= 3.0 else 0.0
    logits = {
        "calm": 1.2 * (1.0 - min(1.0, news_rate)),
        "scheduled_event": 2.5 * near_event,
        "rumor_build": 1.0 * min(1.0, news_rate),
        "live_shock": 2.0 * min(1.0, abs(shock)) + 0.5 * min(1.0, news_rate),
        "post_event_decay": 1.5 * min(1.0, decay),
        "macro_risk": 1.0 * near_event + 0.5 * min(1.0, news_rate),
        "narrative_rotation": 0.8 * min(1.0, news_rate) * (1.0 - min(1.0, abs(shock))),
    }
    return ensure_simplex(softmax(logits), NEWS_SUBREGIMES)


def industry_from_features(features: Mapping[str, float]) -> dict[str, float]:
    beta_stab = _f(features, "beta_stability", 0.5)
    div_z = _f(features, "stock_index_divergence_z", 0.0)
    beta_z = _f(features, "rolling_beta_z", 0.0)
    rel_strength = _f(features, "sector_rs_percentile", 0.5)

    logits = {
        "strong_index_lock": beta_stab * 2.0 + (1.0 - min(1.0, abs(div_z))),
        "strong_sector_lock": beta_stab * 1.5 + rel_strength,
        "local_idiosyncratic": min(1.0, abs(div_z)) * 2.5,
        "beta_expansion": max(0.0, beta_z),
        "beta_compression": max(0.0, -beta_z),
        "sector_rotation": abs(rel_strength - 0.5) * 3.0,
        "divergence_reversion_setup": min(1.0, abs(div_z)) * 2.0 + 0.5 * (1.0 - beta_stab),
    }
    return ensure_simplex(softmax(logits), INDUSTRY_SUBREGIMES)


def build_regime_tensor_from_features(features: Mapping[str, float]) -> RegimeTensor:
    return RegimeTensor(
        indicator=indicator_from_features(features),
        liquidity=liquidity_from_features(features),
        greek=greek_from_features(features),
        news=news_from_features(features),
        industry=industry_from_features(features),
    )


def neutral_regime_tensor() -> RegimeTensor:
    return RegimeTensor(
        indicator=uniform_simplex(INDICATOR_SUBREGIMES),
        liquidity=uniform_simplex(LIQUIDITY_SUBREGIMES),
        greek=uniform_simplex(GREEK_SUBREGIMES),
        news=uniform_simplex(NEWS_SUBREGIMES),
        industry=uniform_simplex(INDUSTRY_SUBREGIMES),
    )
