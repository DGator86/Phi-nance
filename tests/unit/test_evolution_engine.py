"""
tests/unit/test_evolution_engine.py
=====================================

Comprehensive unit tests for phinance.agents.evolution_engine.

Covers:
  - _fitness function (normal, edge cases)
  - Individual dataclass (creation, to_dict, repr)
  - GenerationResult dataclass (creation, to_dict, repr)
  - EvolutionConfig dataclass (defaults, to_dict)
  - EvolutionEngine.__init__ (attributes, defaults)
  - EvolutionEngine._init_population (size, structure)
  - EvolutionEngine._evaluate (fitness computed, no crash)
  - EvolutionEngine._blend_signals (returns ndarray in [-1,1])
  - EvolutionEngine._tournament_select (returns Individual)
  - EvolutionEngine._mutate (returns new Individual, different id)
  - EvolutionEngine._crossover (returns new Individual)
  - EvolutionEngine._evolve (returns new population of same size)
  - EvolutionEngine.run_once (returns GenerationResult)
  - EvolutionEngine.run (history length, best_individual set)
  - EvolutionEngine.evolution_summary (keys)
  - run_evolution convenience function
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phinance.agents.evolution_engine import (
    Individual,
    GenerationResult,
    EvolutionConfig,
    EvolutionEngine,
    _fitness,
    run_evolution,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.002, n)),
            "high":   close * (1 + abs(rng.normal(0, 0.005, n))),
            "low":    close * (1 - abs(rng.normal(0, 0.005, n))),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


DF_200 = make_ohlcv(200)


def make_engine(n_pop: int = 4, n_gen: int = 2) -> EvolutionEngine:
    cfg = EvolutionConfig(
        population_size=n_pop,
        num_generations=n_gen,
        dry_run=True,
        random_seed=7,
        deploy_threshold=999.0,   # never auto-deploy in unit tests
    )
    return EvolutionEngine(ohlcv=DF_200, config=cfg)


# ── _fitness ─────────────────────────────────────────────────────────────────


class TestFitness:
    def test_positive_fitness(self):
        f = _fitness(1.5, 0.1, 20)
        assert f > 0

    def test_zero_sharpe(self):
        assert _fitness(0.0, 0.1, 10) == 0.0

    def test_negative_sharpe(self):
        assert _fitness(-0.5, 0.1, 10) == 0.0

    def test_full_drawdown(self):
        assert _fitness(1.0, 1.0, 10) == 0.0

    def test_zero_trades_nonzero(self):
        # num_trades=0 → trades+1 = 1 → fitness = sharpe*(1-dd)*1
        f = _fitness(1.0, 0.0, 0)
        assert f > 0

    def test_monotone_in_sharpe(self):
        f1 = _fitness(1.0, 0.1, 10)
        f2 = _fitness(2.0, 0.1, 10)
        assert f2 > f1

    def test_monotone_in_trades(self):
        f1 = _fitness(1.0, 0.1, 5)
        f2 = _fitness(1.0, 0.1, 20)
        assert f2 > f1

    def test_monotone_in_drawdown(self):
        f1 = _fitness(1.0, 0.05, 10)
        f2 = _fitness(1.0, 0.50, 10)
        assert f1 > f2

    def test_returns_float(self):
        assert isinstance(_fitness(1.0, 0.1, 5), float)


# ── Individual ────────────────────────────────────────────────────────────────


class TestIndividual:
    def test_default_creation(self):
        ind = Individual()
        assert isinstance(ind.individual_id, str)
        assert len(ind.indicators) == 0
        assert len(ind.weights) == 0
        assert ind.fitness == 0.0
        assert ind.generation == 0

    def test_custom_creation(self):
        ind = Individual(
            indicators=["RSI", "MACD"],
            weights={"RSI": 0.6, "MACD": 0.4},
            fitness=1.23,
            generation=3,
        )
        assert ind.indicators == ["RSI", "MACD"]
        assert ind.weights["RSI"] == 0.6
        assert ind.fitness == 1.23
        assert ind.generation == 3

    def test_to_dict_keys(self):
        ind = Individual(indicators=["RSI"])
        d = ind.to_dict()
        for key in ("individual_id", "indicators", "weights", "blend_method",
                    "fitness", "sharpe", "max_drawdown", "win_rate",
                    "num_trades", "total_return", "generation"):
            assert key in d

    def test_to_dict_indicators(self):
        ind = Individual(indicators=["RSI", "MACD"])
        assert ind.to_dict()["indicators"] == ["RSI", "MACD"]

    def test_repr_contains_class_name(self):
        ind = Individual(indicators=["RSI"], fitness=0.5)
        r = repr(ind)
        assert "Individual" in r
        assert "0.500" in r

    def test_unique_ids(self):
        ids = {Individual().individual_id for _ in range(50)}
        assert len(ids) == 50


# ── GenerationResult ──────────────────────────────────────────────────────────


class TestGenerationResult:
    def test_default_creation(self):
        gr = GenerationResult(generation=1, population_size=10, best_fitness=2.0, mean_fitness=1.0)
        assert gr.generation == 1
        assert gr.population_size == 10
        assert gr.best_fitness == 2.0
        assert not gr.deployed
        assert gr.deployment_id is None

    def test_to_dict_keys(self):
        gr = GenerationResult(generation=0, population_size=5, best_fitness=0.5, mean_fitness=0.2)
        d = gr.to_dict()
        for k in ("generation", "population_size", "best_fitness", "mean_fitness",
                  "deployed", "deployment_id", "elapsed_ms"):
            assert k in d

    def test_repr(self):
        gr = GenerationResult(generation=2, population_size=8, best_fitness=1.5, mean_fitness=0.8)
        assert "GenerationResult" in repr(gr)
        assert "gen=2" in repr(gr)

    def test_deployed_flag(self):
        gr = GenerationResult(generation=0, population_size=4, best_fitness=0.5,
                              mean_fitness=0.3, deployed=True, deployment_id="abc123")
        assert gr.deployed
        assert gr.deployment_id == "abc123"


# ── EvolutionConfig ───────────────────────────────────────────────────────────


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.population_size == 10
        assert cfg.num_generations == 5
        assert cfg.tournament_k == 3
        assert cfg.dry_run is True
        assert cfg.random_seed is None

    def test_custom(self):
        cfg = EvolutionConfig(population_size=20, num_generations=10, dry_run=False)
        assert cfg.population_size == 20
        assert cfg.num_generations == 10
        assert not cfg.dry_run

    def test_to_dict_keys(self):
        d = EvolutionConfig().to_dict()
        for k in ("population_size", "num_generations", "tournament_k",
                  "mutation_rate", "crossover_rate", "dry_run"):
            assert k in d


# ── EvolutionEngine.__init__ ──────────────────────────────────────────────────


class TestEvolutionEngineInit:
    def test_attributes_set(self):
        engine = make_engine()
        assert engine.ohlcv is DF_200
        assert isinstance(engine.config, EvolutionConfig)
        assert engine.best_individual is None
        assert engine.history == []

    def test_all_names_populated(self):
        engine = make_engine()
        assert len(engine._all_names) > 0

    def test_rng_seeded(self):
        e1 = make_engine()
        e2 = make_engine()
        # Both should produce same sequence from seed
        r1 = e1._rng.random()
        r2 = e2._rng.random()
        assert r1 == r2


# ── _init_population ─────────────────────────────────────────────────────────


class TestInitPopulation:
    def test_size(self):
        engine = make_engine(n_pop=6)
        pop = engine._init_population(6)
        assert len(pop) == 6

    def test_all_individuals(self):
        engine = make_engine()
        pop = engine._init_population(4)
        assert all(isinstance(ind, Individual) for ind in pop)

    def test_each_has_indicators(self):
        engine = make_engine()
        pop = engine._init_population(5)
        for ind in pop:
            assert len(ind.indicators) >= engine.config.min_indicators

    def test_weights_normalised(self):
        engine = make_engine()
        pop = engine._init_population(5)
        for ind in pop:
            total = sum(ind.weights.values())
            assert abs(total - 1.0) < 1e-9

    def test_unique_ids(self):
        engine = make_engine()
        pop = engine._init_population(10)
        ids = {ind.individual_id for ind in pop}
        assert len(ids) == 10


# ── _evaluate ─────────────────────────────────────────────────────────────────


class TestEvaluate:
    def test_fitness_set(self):
        engine = make_engine()
        pop = engine._init_population(3)
        pop = engine._evaluate(pop, 0)
        for ind in pop:
            assert isinstance(ind.fitness, float)

    def test_generation_set(self):
        engine = make_engine()
        pop = engine._init_population(3)
        pop = engine._evaluate(pop, 5)
        assert all(ind.generation == 5 for ind in pop)

    def test_no_negative_fitness(self):
        engine = make_engine()
        pop = engine._init_population(4)
        pop = engine._evaluate(pop, 0)
        assert all(ind.fitness >= 0.0 for ind in pop)


# ── _blend_signals ────────────────────────────────────────────────────────────


class TestBlendSignals:
    def test_returns_ndarray(self):
        engine = make_engine()
        ind = Individual(indicators=["RSI"], weights={"RSI": 1.0})
        arr = engine._blend_signals(ind)
        assert isinstance(arr, np.ndarray)

    def test_length_matches_ohlcv(self):
        engine = make_engine()
        ind = Individual(indicators=["EMA Cross"], weights={"EMA Cross": 1.0})
        arr = engine._blend_signals(ind)
        assert len(arr) == len(DF_200)

    def test_values_in_range(self):
        engine = make_engine()
        ind = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.5, "MACD": 0.5})
        arr = engine._blend_signals(ind)
        assert arr.min() >= -1.0 - 1e-9
        assert arr.max() <= 1.0 + 1e-9

    def test_empty_indicators(self):
        engine = make_engine()
        ind = Individual(indicators=[], weights={})
        arr = engine._blend_signals(ind)
        assert np.all(arr == 0.0)


# ── _tournament_select ────────────────────────────────────────────────────────


class TestTournamentSelect:
    def test_returns_individual(self):
        engine = make_engine()
        pop = engine._init_population(6)
        selected = engine._tournament_select(pop)
        assert isinstance(selected, Individual)

    def test_selected_in_population(self):
        engine = make_engine()
        pop = engine._init_population(6)
        for ind in pop:
            ind.fitness = engine._rng.random()
        selected = engine._tournament_select(pop)
        assert selected in pop

    def test_selects_best_when_k_equals_size(self):
        engine = make_engine()
        engine.config.tournament_k = 5
        pop = engine._init_population(5)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        selected = engine._tournament_select(pop)
        assert selected.fitness == max(ind.fitness for ind in pop)


# ── _mutate ───────────────────────────────────────────────────────────────────


class TestMutate:
    def test_returns_individual(self):
        engine = make_engine()
        ind = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.5, "MACD": 0.5})
        child = engine._mutate(ind)
        assert isinstance(child, Individual)

    def test_different_id(self):
        engine = make_engine()
        ind = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.5, "MACD": 0.5})
        child = engine._mutate(ind)
        assert child.individual_id != ind.individual_id

    def test_weights_still_normalised(self):
        engine = make_engine()
        ind = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.5, "MACD": 0.5})
        child = engine._mutate(ind)
        total = sum(child.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_parent_unchanged(self):
        engine = make_engine()
        ind = Individual(
            indicators=["RSI", "MACD"],
            weights={"RSI": 0.5, "MACD": 0.5},
            fitness=1.0,
        )
        _ = engine._mutate(ind)
        assert ind.fitness == 1.0
        assert ind.individual_id  # still set


# ── _crossover ────────────────────────────────────────────────────────────────


class TestCrossover:
    def test_returns_individual(self):
        engine = make_engine()
        pa = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.4, "MACD": 0.6})
        pb = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.7, "MACD": 0.3})
        child = engine._crossover(pa, pb)
        assert isinstance(child, Individual)

    def test_different_id(self):
        engine = make_engine()
        pa = Individual(indicators=["RSI"], weights={"RSI": 1.0})
        pb = Individual(indicators=["RSI"], weights={"RSI": 1.0})
        child = engine._crossover(pa, pb)
        assert child.individual_id != pa.individual_id

    def test_weights_normalised(self):
        engine = make_engine()
        pa = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.4, "MACD": 0.6})
        pb = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.7, "MACD": 0.3})
        child = engine._crossover(pa, pb)
        total = sum(child.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_child_inherits_indicators(self):
        engine = make_engine()
        pa = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.5, "MACD": 0.5})
        pb = Individual(indicators=["RSI", "MACD"], weights={"RSI": 0.3, "MACD": 0.7})
        child = engine._crossover(pa, pb)
        assert set(child.indicators) == {"RSI", "MACD"}


# ── _evolve ───────────────────────────────────────────────────────────────────


class TestEvolve:
    def test_same_size(self):
        engine = make_engine(n_pop=6)
        pop = engine._init_population(6)
        new_pop = engine._evolve(pop)
        assert len(new_pop) == 6

    def test_returns_individuals(self):
        engine = make_engine(n_pop=4)
        pop = engine._init_population(4)
        new_pop = engine._evolve(pop)
        assert all(isinstance(ind, Individual) for ind in new_pop)

    def test_elite_preserved(self):
        engine = make_engine(n_pop=6)
        pop = engine._init_population(6)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best_two = {pop[0].individual_id, pop[1].individual_id}
        new_pop = engine._evolve(pop)
        new_ids = {ind.individual_id for ind in new_pop}
        # Elite should be carried forward
        assert best_two.issubset(new_ids)


# ── run_once ──────────────────────────────────────────────────────────────────


class TestRunOnce:
    def test_returns_generation_result(self):
        engine = make_engine()
        gr = engine.run_once()
        assert isinstance(gr, GenerationResult)

    def test_generation_zero(self):
        engine = make_engine()
        gr = engine.run_once(generation=0)
        assert gr.generation == 0

    def test_population_size_field(self):
        engine = make_engine(n_pop=5)
        gr = engine.run_once()
        assert gr.population_size == 5

    def test_best_individual_set(self):
        engine = make_engine()
        engine.run_once()
        assert engine.best_individual is not None

    def test_history_grows(self):
        engine = make_engine()
        engine.run_once()
        assert len(engine.history) == 1

    def test_mean_fitness_ge_zero(self):
        engine = make_engine()
        gr = engine.run_once()
        assert gr.mean_fitness >= 0.0

    def test_best_fitness_ge_mean(self):
        engine = make_engine()
        gr = engine.run_once()
        assert gr.best_fitness >= gr.mean_fitness - 1e-9


# ── run ───────────────────────────────────────────────────────────────────────


class TestRun:
    def test_history_length(self):
        engine = make_engine(n_pop=4, n_gen=3)
        history = engine.run()
        assert len(history) == 3

    def test_returns_list(self):
        engine = make_engine()
        history = engine.run()
        assert isinstance(history, list)

    def test_all_generation_results(self):
        engine = make_engine()
        history = engine.run()
        assert all(isinstance(gr, GenerationResult) for gr in history)

    def test_best_individual_is_set(self):
        engine = make_engine()
        engine.run()
        assert engine.best_individual is not None
        assert isinstance(engine.best_individual, Individual)

    def test_generations_ordered(self):
        engine = make_engine()
        history = engine.run()
        gens = [gr.generation for gr in history]
        assert gens == sorted(gens)

    def test_single_generation(self):
        cfg = EvolutionConfig(population_size=3, num_generations=1, dry_run=True, random_seed=0)
        engine = EvolutionEngine(ohlcv=DF_200, config=cfg)
        history = engine.run()
        assert len(history) == 1


# ── evolution_summary ─────────────────────────────────────────────────────────


class TestEvolutionSummary:
    def test_empty_history(self):
        engine = make_engine()
        s = engine.evolution_summary
        assert s["generations"] == 0
        assert s["best_fitness"] == 0.0

    def test_after_run_keys(self):
        engine = make_engine()
        engine.run()
        s = engine.evolution_summary
        for k in ("generations", "best_fitness", "final_mean_fitness", "total_deployments"):
            assert k in s

    def test_generations_count(self):
        engine = make_engine(n_gen=2)
        engine.run()
        assert engine.evolution_summary["generations"] == 2

    def test_best_fitness_non_negative(self):
        engine = make_engine()
        engine.run()
        assert engine.evolution_summary["best_fitness"] >= 0.0


# ── run_evolution ─────────────────────────────────────────────────────────────


class TestRunEvolution:
    def test_returns_tuple(self):
        result = run_evolution(DF_200, population_size=3, num_generations=1, random_seed=1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_history_not_empty(self):
        history, best = run_evolution(DF_200, population_size=3, num_generations=1, random_seed=1)
        assert len(history) >= 1

    def test_best_individual_type(self):
        _, best = run_evolution(DF_200, population_size=3, num_generations=1, random_seed=1)
        assert isinstance(best, Individual)

    def test_dry_run_default(self):
        history, _ = run_evolution(DF_200, population_size=3, num_generations=1)
        assert all(not gr.deployed for gr in history)

    def test_deterministic_with_seed(self):
        h1, b1 = run_evolution(DF_200, population_size=4, num_generations=1, random_seed=99)
        h2, b2 = run_evolution(DF_200, population_size=4, num_generations=1, random_seed=99)
        assert b1.indicators == b2.indicators
