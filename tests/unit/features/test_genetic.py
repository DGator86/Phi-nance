import numpy as np
import pandas as pd

from phinance.features.genetic_features import GPFeatureConfig, GPFeatureDiscovery, evaluate_expression


def _frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    volume = rng.integers(1_000, 3_000, size=n)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": volume,
        }
    )


def test_gp_feature_discovery_returns_expressions():
    frame = _frame()
    discovery = GPFeatureDiscovery(frame, config=GPFeatureConfig(population_size=8, generations=2, top_k=3, random_seed=1))
    results = discovery.evolve()

    assert results
    assert len(results) <= 3
    assert "expression" in results[0]
    assert "fitness" in results[0]


def test_expression_evaluation():
    frame = _frame(10)
    val = evaluate_expression("add(close, volume)", frame)
    assert isinstance(val, float)
