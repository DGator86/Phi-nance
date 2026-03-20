from __future__ import annotations

import pytest


def _deps():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    return np, pd


def test_hilbert_transform_sine_phase_shift() -> None:
    np, pd = _deps()
    from phi.mft.complex import hilbert_transform

    n = 512
    x = np.linspace(0, 8 * np.pi, n)
    idx = pd.RangeIndex(n)
    sine = pd.Series(np.sin(x), index=idx)

    imag = hilbert_transform(sine)
    corr = np.corrcoef(imag.to_numpy(), (-np.cos(x)))[0, 1]
    assert corr > 0.98


def test_complex_potential_sine_amplitude_and_phase_change() -> None:
    np, pd = _deps()
    from phi.mft.complex import complex_potential

    n = 512
    x = np.linspace(0, 6 * np.pi, n)
    sine = pd.Series(np.sin(x), index=pd.RangeIndex(n))
    cp = complex_potential(sine)

    amp = cp["amplitude"].iloc[50:-50]
    np.testing.assert_almost_equal(float(amp.mean()), 1.0, decimal=2)
    assert float(cp["phase_change"].iloc[50:-50].std()) < 0.05


def test_volume_price_interaction_positive_when_aligned() -> None:
    np, pd = _deps()
    from phi.mft.volume_field import volume_price_interaction

    n = 200
    idx = pd.RangeIndex(n)
    price = pd.Series(np.linspace(100.0, 130.0, n) + 0.5 * np.sin(np.linspace(0, 8, n)), index=idx)
    volume = pd.Series(np.linspace(1000.0, 2000.0, n), index=idx)

    interaction = volume_price_interaction(price, volume, sigma=6.0, corr_window=20)
    assert interaction.iloc[-40:].mean() > 0


def test_rolling_spectral_power_detects_low_frequency_signal() -> None:
    np, pd = _deps()
    from phi.mft.fourier import rolling_spectral_power

    n = 512
    t = np.arange(n)
    low_freq = np.sin(2 * np.pi * 0.05 * t)
    series = pd.Series(low_freq, index=pd.RangeIndex(n))

    bands = [(0.0, 0.2), (0.2, 1.0)]
    power = rolling_spectral_power(series, window=128, bands=bands)

    tail = power.iloc[-50:]
    assert tail["band_0"].mean() > 0.7
    assert tail["band_1"].mean() < 0.3


def test_complex_potential_preserves_nan_gaps_in_phase_change() -> None:
    np, pd = _deps()
    from phi.mft.complex import complex_potential

    series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 120)), index=pd.RangeIndex(120))
    series.iloc[60] = np.nan

    cp = complex_potential(series, hilbert_window=20)
    assert np.isnan(cp.loc[60, "phase_change"])
    assert cp.loc[40, "amplitude"] > 0.0
    assert cp.loc[90, "amplitude"] > 0.0


def test_rolling_spectral_power_includes_nyquist_upper_band() -> None:
    np, pd = _deps()
    from phi.mft.fourier import rolling_spectral_power

    n = 256
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    series = pd.Series(signal, index=pd.RangeIndex(n))

    power = rolling_spectral_power(series, window=64, bands=[(0.0, 0.5), (0.5, 1.0)])
    tail = power.iloc[-30:]

    assert tail["band_1"].mean() > 0.95
    assert tail["band_0"].mean() < 0.05
