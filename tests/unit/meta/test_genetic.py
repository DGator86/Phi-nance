import numpy as np
import pandas as pd

from phinance.meta.genetic import GPConfig, GeneticStrategySearch


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    base = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )


def test_genetic_search_returns_best_strategies():
    df = _synthetic_ohlcv()
    search = GeneticStrategySearch(
        ohlcv=df,
        config=GPConfig(population_size=8, generations=2, top_k=3, random_seed=1),
    )
    result = search.evolve()

    assert "best_strategies" in result
    assert len(result["best_strategies"]) > 0
    assert "expression" in result["best_strategies"][0]
