from __future__ import annotations

import numpy as np
import pandas as pd

from phi.indicators.orderflow.providers.ohlcv_provider import OHLCVOrderFlowProvider
from phi.indicators.orderflow.vwap import compute_vwap_series, compute_vwap_signal
from phi.indicators.orderflow.volume_profile import compute_volume_profile_signal
from phi.indicators.orderflow.cumulative_delta import compute_cumulative_delta_signal
from phi.indicators.orderflow.liquidity import compute_liquidity_signal
from phi.indicators.registry import compute_signal, get_indicator


def _sample_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(100.0, 110.0, n)
    open_ = base + np.sin(np.arange(n) / 5.0)
    close = open_ + np.cos(np.arange(n) / 7.0)
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = np.linspace(1000.0, 3000.0, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_ohlcv_orderflow_provider_directional_split() -> None:
    df = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [11.0, 9.0, 10.0],
            "volume": [100.0, 120.0, 80.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    flow = OHLCVOrderFlowProvider().get_order_flow(df)

    assert flow.loc[df.index[0], "buy_volume"] == 100.0
    assert flow.loc[df.index[0], "sell_volume"] == 0.0
    assert flow.loc[df.index[1], "buy_volume"] == 0.0
    assert flow.loc[df.index[1], "sell_volume"] == 120.0
    assert flow.loc[df.index[2], "buy_volume"] == 40.0
    assert flow.loc[df.index[2], "sell_volume"] == 40.0


def test_vwap_outputs_align_and_signal_bounds() -> None:
    df = _sample_ohlcv()
    vwap = compute_vwap_series(df)
    signal = compute_vwap_signal(df, atr_period=14, clip_value=2.0)

    assert list(vwap.index) == list(df.index)
    assert list(signal.index) == list(df.index)
    assert signal.min() >= -1.0
    assert signal.max() <= 1.0


def test_volume_profile_outputs() -> None:
    df = _sample_ohlcv()
    poc, signal = compute_volume_profile_signal(df, window=20, bins=10)

    assert len(poc) == len(df)
    assert len(signal) == len(df)
    assert set(signal.dropna().unique()).issubset({-1.0, 1.0})


def test_cumulative_delta_and_liquidity_bounds() -> None:
    df = _sample_ohlcv()
    flow = OHLCVOrderFlowProvider().get_order_flow(df)

    cd = compute_cumulative_delta_signal(flow, df["volume"], window=20, clip_value=1.0)
    liq = compute_liquidity_signal(df, flow, window=20)

    assert cd.min() >= -1.0
    assert cd.max() <= 1.0
    assert liq.min() >= -1.0
    assert liq.max() <= 1.0


def test_registry_has_orderflow_indicators() -> None:
    df = _sample_ohlcv()
    for name in ["orderflow_vwap", "volume_profile", "cumulative_delta", "liquidity"]:
        entry = get_indicator(name)
        assert entry is not None
        out = compute_signal(name, df)
        assert len(out) == len(df)
        assert out.min() >= -1.0
        assert out.max() <= 1.0
