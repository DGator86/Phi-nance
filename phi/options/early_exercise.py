"""Simple early-exercise heuristics for American options."""

from __future__ import annotations


def should_exercise_early(option_type: str, spot: float, strike: float, time_to_expiry: float, dividend_yield: float = 0.0) -> bool:
    """Heuristic early exercise signal.

    - Calls on non-dividend underlyings are never exercised early.
    - Puts are exercised early when ITM and close to expiry.
    - Calls may be exercised near expiry when dividends are significant.
    """
    option_type = option_type.lower()
    if option_type == "call" and dividend_yield <= 0:
        return False

    itm = (spot > strike) if option_type == "call" else (spot < strike)
    near_expiry = time_to_expiry <= (30.0 / 365.0)
    if option_type == "put":
        return bool(itm and near_expiry)
    return bool(itm and near_expiry and dividend_yield > 0)
