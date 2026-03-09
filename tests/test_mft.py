from __future__ import annotations

import pytest


def _deps():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    return np, pd


def test_field_potential_preserves_shape_and_index() -> None:
    np, pd = _deps()
    from phi.mft.field import field_potential

    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    potential = field_potential(close, kernel="gaussian", sigma=8.0)

    assert isinstance(potential, pd.Series)
    assert len(potential) == len(close)
    assert potential.index.equals(close.index)
    assert potential.notna().all()


def test_compute_field_dynamics_columns_and_lengths() -> None:
    np, pd = _deps()
    from phi.mft.field import compute_field_dynamics

    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    close = pd.Series(100.0 + np.sin(np.linspace(0, 6, len(idx))), index=idx)

    dynamics = compute_field_dynamics(close, kernel="exp", sigma=6.0)

    assert {"potential", "gradient", "laplacian", "energy"}.issubset(dynamics.columns)
    assert len(dynamics) == len(close)
    assert dynamics.index.equals(close.index)


def test_mft_signal_detects_direction_on_monotonic_series() -> None:
    np, pd = _deps()
    from phi.mft.signals import mft_signal

    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    up_close = pd.Series(np.linspace(10.0, 30.0, len(idx)), index=idx)
    down_close = pd.Series(np.linspace(30.0, 10.0, len(idx)), index=idx)

    up_signal = mft_signal(up_close, kernel="gaussian", sigma=4.0, threshold=0.0, smooth_window=1)
    down_signal = mft_signal(down_close, kernel="gaussian", sigma=4.0, threshold=0.0, smooth_window=1)

    assert up_signal.iloc[-10:].mean() > 0
    assert down_signal.iloc[-10:].mean() < 0


def test_mft_energy_signal_bounded_range() -> None:
    np, pd = _deps()
    from phi.mft.signals import mft_energy_signal

    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    close = pd.Series(100 + np.sin(np.linspace(0, 4 * np.pi, len(idx))), index=idx)

    signal = mft_energy_signal(close, sigma=5.0, energy_window=15)

    assert signal.between(-1.0, 1.0).all()


def test_mft_backward_compat_aliases() -> None:
    np, pd = _deps()
    from phi.indicators.registry import compute_signal
    from phi.indicators.simple import compute_indicator

    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    close = np.linspace(100.0, 130.0, len(idx))
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )

    legacy_registry_signal = compute_signal("phi_mft", ohlcv)
    assert len(legacy_registry_signal) == len(ohlcv)

    legacy_simple_signal = compute_indicator("Phi-Bot (MFT)", ohlcv, {})
    assert len(legacy_simple_signal) == len(ohlcv)
