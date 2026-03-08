from __future__ import annotations

import pytest


def _deps():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    return np, pd


def _ohlcv(rows: int = 180):
    np, pd = _deps()
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, rows)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, rows)),
            "high": close * (1 + np.abs(rng.normal(0.003, 0.002, rows))),
            "low": close * (1 - np.abs(rng.normal(0.003, 0.002, rows))),
            "close": close,
            "volume": rng.integers(100_000, 500_000, rows),
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "name,params",
    [
        ("Rolling Entropy", {"window": 20, "bins": 10}),
        ("Mutual Information", {"window": 30, "bins": 8, "lag": 1}),
        ("Fisher Information", {"window": 20}),
        ("KL Divergence", {"window": 30, "bins": 10}),
    ],
)
def test_information_indicators_compute_from_simple_layer(name: str, params: dict) -> None:
    pd = pytest.importorskip("pandas")
    from phi.indicators.simple import compute_indicator

    out = compute_indicator(name, _ohlcv(), params)
    assert isinstance(out, pd.Series)
    assert len(out) > 0
    assert out.index.equals(_ohlcv().index)


@pytest.mark.parametrize(
    "name",
    ["rolling_entropy", "mutual_information", "fisher_information", "kl_divergence"],
)
def test_information_indicators_registered_in_registry(name: str) -> None:
    pd = pytest.importorskip("pandas")
    from phi.indicators.registry import compute_signal, get_indicator

    spec = get_indicator(name)
    assert spec is not None
    out = compute_signal(name, _ohlcv())
    assert isinstance(out, pd.Series)
    assert len(out) == len(_ohlcv())
