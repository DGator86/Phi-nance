from __future__ import annotations

import pandas as pd

from phi.options.iv_surface import IVSurface


def test_iv_surface_interpolates_midpoint():
    chain = pd.DataFrame(
        {
            "strike": [90, 100, 110, 90, 100, 110],
            "expiry": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3],
            "iv": [0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
        }
    )
    surface = IVSurface(chain)
    iv = surface.get_iv(100, 0.2)
    assert 0.20 <= iv <= 0.23


def test_iv_surface_clamps_out_of_range():
    chain = pd.DataFrame({"strike": [100, 105, 110], "expiry": [0.1, 0.2, 0.3], "iv": [0.2, 0.21, 0.22]})
    surface = IVSurface(chain)
    iv = surface.get_iv(1000, 5.0)
    assert iv > 0
