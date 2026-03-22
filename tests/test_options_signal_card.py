"""Options signal card generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.options.signal_card import OptionsSignalCard
from phi.options.signal_generator import build_options_signal_card


def _ohlcv(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.4, size=n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": rng.integers(1_000_000, 4_000_000, size=n),
        },
        index=idx,
    )


def test_build_options_signal_card_returns_model() -> None:
    card = build_options_signal_card(_ohlcv(), symbol="ZZZ")
    assert isinstance(card, OptionsSignalCard)
    assert card.symbol == "ZZZ"
    assert card.action in ("ENTER", "WAIT", "SKIP")
    assert card.target_exit_pct == 0.50
    assert card.stop_exit_pct == 1.00
    assert card.composite_regime
    assert isinstance(card.reasoning, list)
