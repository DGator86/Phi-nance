from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from phi.indicators.information_flow import rolling_granger_causality, rolling_transfer_entropy
from phi.indicators.simple import compute_indicator


def _prices_with_lead_lag(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, n)
    y = np.zeros(n)
    noise = rng.normal(0.0, 0.003, n)
    for i in range(1, n):
        y[i] = 0.65 * x[i - 1] + noise[i]

    px = 100 * np.cumprod(1 + x)
    py = 100 * np.cumprod(1 + y)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"AAA": px, "BBB": py}, index=idx)


def test_transfer_entropy_detects_directionality() -> None:
    prices = _prices_with_lead_lag()
    te_ab = rolling_transfer_entropy(prices, from_symbol="AAA", to_symbol="BBB", window=60, bins=3)
    te_ba = rolling_transfer_entropy(prices, from_symbol="BBB", to_symbol="AAA", window=60, bins=3)

    assert te_ab.iloc[-80:].mean() > te_ba.iloc[-80:].mean()
    assert np.isfinite(te_ab.iloc[-1])


def test_granger_causality_detects_directionality() -> None:
    prices = _prices_with_lead_lag()
    p_ab = rolling_granger_causality(prices, from_symbol="AAA", to_symbol="BBB", window=80, maxlags=2, output="pvalue")
    p_ba = rolling_granger_causality(prices, from_symbol="BBB", to_symbol="AAA", window=80, maxlags=2, output="pvalue")

    assert p_ab.iloc[-60:].median() < 0.1
    assert p_ab.iloc[-60:].median() < p_ba.iloc[-60:].median()


def test_information_flow_edge_cases() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    flat = pd.DataFrame({"AAA": np.full(120, 100.0), "BBB": np.full(120, 100.0)}, index=idx)

    te = rolling_transfer_entropy(flat, "AAA", "BBB", window=50, bins=3)
    gc = rolling_granger_causality(flat, "AAA", "BBB", window=50, maxlags=2, output="pvalue")

    assert te.isna().sum() >= 1 or (te.fillna(0.0).abs().max() < 1e-6)
    assert gc.isna().sum() >= 1


def test_simple_wrapper_supports_information_flow() -> None:
    prices = _prices_with_lead_lag()
    out = compute_indicator(
        "Transfer Entropy",
        prices,
        {"from_symbol": "AAA", "to_symbol": "BBB", "window": 50, "bins": 3, "normalize": True},
    )
    assert isinstance(out, pd.Series)
    assert len(out) == len(prices)
