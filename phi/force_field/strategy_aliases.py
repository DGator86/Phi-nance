"""Map force-field snake_case keys to legacy ``phi.regime.strategy_mapping`` labels."""

from __future__ import annotations

FORCE_FIELD_TO_LEGACY_NAME: dict[str, str] = {
    "call_debit_spread": "Bull Call Spread",
    "put_debit_spread": "Bear Put Spread",
    "call_credit_spread": "Bear Call Spread",
    "put_credit_spread": "Bull Put Spread",
    "iron_condor": "Iron Condor",
    "iron_butterfly": "Iron Butterfly",
    "long_straddle": "Long Straddle",
    "long_strangle": "Long Strangle",
    "calendar": "Calendar Spread",
    "diagonal": "Diagonal Spread",
    "long_call_butterfly": "Long Call Butterfly",
    "long_put_butterfly": "Long Put Butterfly",
    "broken_wing_butterfly": "Diagonal Spread",
    "call_backspread": "Bull Call Spread",
    "put_backspread": "Bear Put Spread",
}

LEGACY_NAME_TO_FORCE_FIELD: dict[str, str] = {}
for _ff_key, _legacy in FORCE_FIELD_TO_LEGACY_NAME.items():
    LEGACY_NAME_TO_FORCE_FIELD.setdefault(_legacy, _ff_key)


def to_legacy_strategy_name(force_field_key: str) -> str | None:
    return FORCE_FIELD_TO_LEGACY_NAME.get(force_field_key)
