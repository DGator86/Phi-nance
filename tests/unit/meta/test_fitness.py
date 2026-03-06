import numpy as np
import pandas as pd
from deap import base, creator, gp

from phinance.meta.fitness import GPFitnessEvaluator
from phinance.meta.primitives import build_feature_frame, build_primitive_set


def _synthetic_ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(11).normal(0, 0.5, size=n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1000 + np.arange(n),
        },
        index=idx,
    )


def test_fitness_evaluator_caches_results():
    df = _synthetic_ohlcv()
    features = build_feature_frame(df)
    pset, context = build_primitive_set(features.columns.tolist())

    if not hasattr(creator, "FitnessMaxMetaTest"):
        creator.create("FitnessMaxMetaTest", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualMetaTest"):
        creator.create("IndividualMetaTest", gp.PrimitiveTree, fitness=creator.FitnessMaxMetaTest)

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)

    individual = creator.IndividualMetaTest.from_string("tanh(close)", pset)
    evaluator = GPFitnessEvaluator(df, context)

    first = evaluator.evaluate(individual, toolbox)
    second = evaluator.evaluate(individual, toolbox)

    assert first == second
    assert len(evaluator.cache) == 1
