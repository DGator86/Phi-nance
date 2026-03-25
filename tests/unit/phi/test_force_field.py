"""Tests for phi.force_field regime tensor → strategy pipeline."""

from __future__ import annotations

from phi.force_field import (
    build_regime_tensor_from_features,
    run_pipeline,
    validate_simplex,
)
from phi.force_field.taxonomy import (
    GREEK_SUBREGIMES,
    INDICATOR_SUBREGIMES,
    INDUSTRY_SUBREGIMES,
    LIQUIDITY_SUBREGIMES,
    NEWS_SUBREGIMES,
)
from phi.regime.strategy_mapping import force_field_ranking_to_legacy, is_approved_strategy


def test_regime_tensor_simplexes() -> None:
    features = {
        "adx": 28.0,
        "spread_pct": 0.0004,
        "iv_rank": 0.55,
        "net_gex_proxy": -0.2,
        "headline_velocity": 0.1,
        "stock_index_divergence_z": 0.3,
    }
    R = build_regime_tensor_from_features(features)
    assert validate_simplex(R.indicator, INDICATOR_SUBREGIMES, tol=0.02)
    assert validate_simplex(R.liquidity, LIQUIDITY_SUBREGIMES, tol=0.02)
    assert validate_simplex(R.greek, GREEK_SUBREGIMES, tol=0.02)
    assert validate_simplex(R.news, NEWS_SUBREGIMES, tol=0.02)
    assert validate_simplex(R.industry, INDUSTRY_SUBREGIMES, tol=0.02)


def test_pipeline_ranks_and_legacy_mapping() -> None:
    features = {
        "adx": 18.0,
        "spread_pct": 0.0003,
        "iv_rank": 0.35,
        "net_gex_proxy": 0.5,
        "term_structure_edge": 0.2,
        "market_cap_billions": 200.0,
        "adv_dollar": 1e9,
    }
    _, ranked = run_pipeline(features)
    assert len(ranked) >= 5
    best_non_veto = next(r for r in ranked if r.score > -500)
    assert best_non_veto.strategy_name
    legacy_rows = force_field_ranking_to_legacy(ranked)
    assert legacy_rows[0][0]
    labels = [row[0] for row in legacy_rows if row[1] > -500]
    assert any(is_approved_strategy(s) for s in labels)


def test_potential_gradient_sign() -> None:
    from phi.force_field.potential_field import GaussianWell, neg_gradient_1d

    wells = [GaussianWell(center=100.0, depth=2.0, sigma=1.0)]
    f_left = neg_gradient_1d(99.0, wells)
    f_right = neg_gradient_1d(101.0, wells)
    assert f_left > 0.0
    assert f_right < 0.0
