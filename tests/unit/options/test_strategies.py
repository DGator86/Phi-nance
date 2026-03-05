from __future__ import annotations

import pandas as pd

from phi.options.iv_surface import IVSurface
from phi.options.strategies import SingleLeg, Straddle, VerticalSpread


def _surface() -> IVSurface:
    return IVSurface(pd.DataFrame({"strike": [90, 100, 110, 90, 100, 110], "expiry": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3], "iv": [0.2, 0.2, 0.21, 0.22, 0.22, 0.23]}))


def test_single_leg_structure_and_greeks():
    strat = SingleLeg("call", "buy", 100, 0.25, 2)
    assert len(strat.legs()) == 1
    gs = strat.greeks(100, 0.01, _surface())
    assert "delta" in gs and gs["delta"] > 0


def test_vertical_spread_two_legs():
    strat = VerticalSpread("call", 95, 105, 0.25, 1)
    assert len(strat.legs()) == 2
    premium = strat.net_premium(100, 0.01, _surface())
    assert isinstance(premium, float)


def test_straddle_has_call_and_put():
    strat = Straddle(100, 0.25, 1)
    legs = strat.legs()
    assert {leg.option_type for leg in legs} == {"call", "put"}
