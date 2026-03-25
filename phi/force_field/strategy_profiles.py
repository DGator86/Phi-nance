"""Payoff-shape profiles: match field geometry to defined-risk strategies."""

from __future__ import annotations

# Six axes: directional, expansion, range, pin, term_structure, liquidity appetite
StrategyProfile = dict[str, float]

STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "call_debit_spread": {
        "directional_need": 0.9,
        "expansion_need": 0.5,
        "range_preference": -0.8,
        "pin_preference": -0.4,
        "term_structure_need": 0.1,
        "liquidity_need": 0.6,
    },
    "put_debit_spread": {
        "directional_need": -0.9,
        "expansion_need": 0.5,
        "range_preference": -0.8,
        "pin_preference": -0.4,
        "term_structure_need": 0.1,
        "liquidity_need": 0.6,
    },
    "call_credit_spread": {
        "directional_need": -0.35,
        "expansion_need": -0.65,
        "range_preference": 0.75,
        "pin_preference": 0.55,
        "term_structure_need": 0.15,
        "liquidity_need": 0.75,
    },
    "put_credit_spread": {
        "directional_need": 0.35,
        "expansion_need": -0.65,
        "range_preference": 0.75,
        "pin_preference": 0.55,
        "term_structure_need": 0.15,
        "liquidity_need": 0.75,
    },
    "iron_condor": {
        "directional_need": 0.0,
        "expansion_need": -0.8,
        "range_preference": 0.95,
        "pin_preference": 0.8,
        "term_structure_need": 0.2,
        "liquidity_need": 0.8,
    },
    "iron_butterfly": {
        "directional_need": 0.0,
        "expansion_need": -0.9,
        "range_preference": 0.9,
        "pin_preference": 0.9,
        "term_structure_need": 0.25,
        "liquidity_need": 0.8,
    },
    "long_straddle": {
        "directional_need": 0.0,
        "expansion_need": 1.0,
        "range_preference": -1.0,
        "pin_preference": -0.7,
        "term_structure_need": 0.4,
        "liquidity_need": 0.7,
    },
    "long_strangle": {
        "directional_need": 0.0,
        "expansion_need": 1.0,
        "range_preference": -0.95,
        "pin_preference": -0.65,
        "term_structure_need": 0.35,
        "liquidity_need": 0.65,
    },
    "calendar": {
        "directional_need": 0.1,
        "expansion_need": 0.2,
        "range_preference": 0.4,
        "pin_preference": 0.5,
        "term_structure_need": 1.0,
        "liquidity_need": 0.8,
    },
    "diagonal": {
        "directional_need": 0.35,
        "expansion_need": 0.35,
        "range_preference": 0.2,
        "pin_preference": 0.35,
        "term_structure_need": 0.85,
        "liquidity_need": 0.75,
    },
    "long_call_butterfly": {
        "directional_need": 0.0,
        "expansion_need": -0.2,
        "range_preference": 0.5,
        "pin_preference": 1.0,
        "term_structure_need": 0.2,
        "liquidity_need": 0.75,
    },
    "long_put_butterfly": {
        "directional_need": 0.0,
        "expansion_need": -0.2,
        "range_preference": 0.5,
        "pin_preference": 1.0,
        "term_structure_need": 0.2,
        "liquidity_need": 0.75,
    },
    "broken_wing_butterfly": {
        "directional_need": 0.15,
        "expansion_need": -0.1,
        "range_preference": 0.55,
        "pin_preference": 0.85,
        "term_structure_need": 0.3,
        "liquidity_need": 0.7,
    },
    "call_backspread": {
        "directional_need": 0.45,
        "expansion_need": 0.9,
        "range_preference": -0.5,
        "pin_preference": -0.45,
        "term_structure_need": 0.2,
        "liquidity_need": 0.55,
    },
    "put_backspread": {
        "directional_need": -0.45,
        "expansion_need": 0.9,
        "range_preference": -0.5,
        "pin_preference": -0.45,
        "term_structure_need": 0.2,
        "liquidity_need": 0.55,
    },
}


def list_strategies() -> tuple[str, ...]:
    return tuple(sorted(STRATEGY_PROFILES.keys()))
