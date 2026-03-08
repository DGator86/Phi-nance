from __future__ import annotations

import numpy as np
import pandas as pd

from phi.indicators.information import (
    compute_entropy_signal,
    compute_fisher_information_signal,
    compute_kld_signal,
    compute_mutual_info_signal,
)
from phi.indicators.registry import compute_signal, get_indicator


def _ohlcv_from_returns(returns: np.ndarray, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0 + 0.002)
    low = np.minimum(open_, close) * (1.0 - 0.002)
    volume = 2000.0 + rng.normal(0, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_entropy_low_for_constant_returns_and_higher_for_random() -> None:
    n = 240
    constant_returns = np.full(n, 0.001)
    random_returns = np.random.default_rng(42).normal(0.0, 0.01, n)

    df_const = _ohlcv_from_returns(constant_returns)
    df_rand = _ohlcv_from_returns(random_returns)

    entropy_const = compute_entropy_signal(df_const, window=40, bins=20)
    entropy_rand = compute_entropy_signal(df_rand, window=40, bins=20)

    const_tail = float(entropy_const.iloc[-80:].mean())
    rand_tail = float(entropy_rand.iloc[-80:].mean())
    assert const_tail < -0.5
    assert rand_tail > const_tail


def test_mutual_information_detects_dependency() -> None:
    n = 260
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.008, n)
    df = _ohlcv_from_returns(returns)
    df["volume"] = 1000.0 + np.abs(returns) * 250000.0 + rng.normal(0, 50, n)

    mi = compute_mutual_info_signal(df, window=40, bins=16, mode="price_volume")
    tail_mean = float(mi.iloc[-80:].mean())
    assert tail_mean > -0.2


def test_fisher_information_higher_when_variance_is_lower() -> None:
    n = 320
    rng = np.random.default_rng(10)
    low_var = rng.normal(0.0, 0.002, n // 2)
    hi_var = rng.normal(0.0, 0.02, n // 2)
    fisher = compute_fisher_information_signal(_ohlcv_from_returns(np.concatenate([low_var, hi_var])), window=40)

    pre = float(fisher.iloc[120:150].mean())
    post = float(fisher.iloc[230:280].mean())
    assert pre > post


def test_kld_spikes_when_distribution_changes() -> None:
    n = 400
    rng = np.random.default_rng(999)
    first = rng.normal(0.0, 0.004, n // 2)
    second = rng.normal(0.01, 0.02, n // 2)
    df = _ohlcv_from_returns(np.concatenate([first, second]))

    kld = compute_kld_signal(df, recent_window=30, reference_window=60, bins=20, sigmoid_scale=2.5)

    baseline = float(kld.iloc[120:170].mean())
    transition_peak = float(kld.iloc[185:230].max())
    assert transition_peak > baseline + 0.05


def test_registry_entries_exist_and_output_bounds() -> None:
    df = _ohlcv_from_returns(np.random.default_rng(321).normal(0.0, 0.01, 260))
    names = ["return_entropy", "mutual_information", "fisher_information", "kld_regime_shift"]
    for name in names:
        assert get_indicator(name) is not None
        out = compute_signal(name, df)
        assert len(out) == len(df)
        assert out.min() >= -1.0
        assert out.max() <= 1.0
