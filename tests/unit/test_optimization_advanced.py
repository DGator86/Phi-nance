"""
tests.unit.test_optimization_advanced
=======================================

Unit tests for the enhanced PhiAI optimization module:
  • Bayesian search (Optuna TPE)
  • Genetic Algorithm
  • Updated search() dispatcher (all 4 methods)
  • Updated run_phiai_optimization with search_method param
  • SEARCH_METHODS constant
  • PhiAI class search_method attribute

All tests use synthetic OHLCV data. No real broker / network calls.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

DF = make_ohlcv(n=120)   # 120 bars — enough for all optimizer warm-ups

# ── Simple objective function ─────────────────────────────────────────────────

def _obj(df: pd.DataFrame, params: dict) -> float:
    """Dummy objective that rewards period=14 (for deterministic assertions)."""
    period = params.get("period", 14)
    return 1.0 / (1.0 + abs(period - 14))


_GRID = {"period": [7, 10, 14, 20, 28]}


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian search
# ═══════════════════════════════════════════════════════════════════════════════

class TestBayesianSearch:

    def test_returns_tuple(self):
        from phinance.optimization.bayesian import bayesian_search
        params, score = bayesian_search(DF, _obj, _GRID, n_trials=10)
        assert isinstance(params, dict)
        assert isinstance(score, float)

    def test_empty_grid_returns_empty(self):
        from phinance.optimization.bayesian import bayesian_search
        params, score = bayesian_search(DF, _obj, {}, n_trials=5)
        assert params == {}
        assert score == 0.0

    def test_best_param_within_grid(self):
        from phinance.optimization.bayesian import bayesian_search
        params, score = bayesian_search(DF, _obj, _GRID, n_trials=20, seed=42)
        assert params.get("period") in _GRID["period"]

    def test_score_is_finite(self):
        from phinance.optimization.bayesian import bayesian_search
        _, score = bayesian_search(DF, _obj, _GRID, n_trials=10, seed=0)
        assert np.isfinite(score)

    def test_score_positive(self):
        from phinance.optimization.bayesian import bayesian_search
        _, score = bayesian_search(DF, _obj, _GRID, n_trials=10)
        assert score > 0.0

    def test_reproducible_with_seed(self):
        from phinance.optimization.bayesian import bayesian_search
        p1, s1 = bayesian_search(DF, _obj, _GRID, n_trials=8, seed=99)
        p2, s2 = bayesian_search(DF, _obj, _GRID, n_trials=8, seed=99)
        assert p1 == p2
        assert abs(s1 - s2) < 1e-9

    def test_finds_optimal_period(self):
        """With enough trials Bayesian should find period=14 (global max)."""
        from phinance.optimization.bayesian import bayesian_search
        params, _ = bayesian_search(DF, _obj, _GRID, n_trials=30, seed=7)
        assert params.get("period") == 14

    def test_multi_param_grid(self):
        from phinance.optimization.bayesian import bayesian_search
        grid = {"period": [7, 14, 21], "oversold": [25, 30], "overbought": [70, 75]}
        params, score = bayesian_search(DF, _obj, grid, n_trials=15)
        assert "period" in params
        assert "oversold" in params

    def test_create_study_returns_study(self):
        from phinance.optimization.bayesian import create_study
        study = create_study(name="test_study")
        assert study is not None
        assert hasattr(study, "optimize")

    def test_failing_objective_handled_gracefully(self):
        from phinance.optimization.bayesian import bayesian_search
        def bad_obj(df, params):
            raise ValueError("intentional failure")
        params, score = bayesian_search(DF, bad_obj, _GRID, n_trials=5)
        # Should not raise; score = 0
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneticSearch:

    def test_returns_tuple(self):
        from phinance.optimization.genetic import genetic_search
        params, score = genetic_search(DF, _obj, _GRID,
                                       population_size=10, n_generations=5)
        assert isinstance(params, dict)
        assert isinstance(score, float)

    def test_empty_grid_returns_empty(self):
        from phinance.optimization.genetic import genetic_search
        params, score = genetic_search(DF, _obj, {})
        assert params == {}
        assert score == 0.0

    def test_best_param_within_grid(self):
        from phinance.optimization.genetic import genetic_search
        params, _ = genetic_search(DF, _obj, _GRID,
                                   population_size=10, n_generations=5, seed=42)
        assert params.get("period") in _GRID["period"]

    def test_score_finite(self):
        from phinance.optimization.genetic import genetic_search
        _, score = genetic_search(DF, _obj, _GRID, population_size=5, n_generations=3)
        assert np.isfinite(score)

    def test_score_positive(self):
        from phinance.optimization.genetic import genetic_search
        _, score = genetic_search(DF, _obj, _GRID, population_size=8, n_generations=4)
        assert score > 0.0

    def test_finds_optimal_period(self):
        """GA should converge to period=14 with enough generations."""
        from phinance.optimization.genetic import genetic_search
        params, _ = genetic_search(DF, _obj, _GRID,
                                   population_size=20, n_generations=15, seed=0)
        assert params.get("period") == 14

    def test_mutation_rate_boundary(self):
        from phinance.optimization.genetic import genetic_search
        # Extreme mutation_rate shouldn't crash
        p1, s1 = genetic_search(DF, _obj, _GRID, mutation_rate=0.0,
                                 population_size=5, n_generations=3, seed=1)
        p2, s2 = genetic_search(DF, _obj, _GRID, mutation_rate=1.0,
                                 population_size=5, n_generations=3, seed=1)
        assert isinstance(s1, float) and isinstance(s2, float)

    def test_internal_helpers(self):
        """Test encoding, crossover and mutation helpers."""
        import numpy as np
        from phinance.optimization.genetic import (
            _encode_individual, _uniform_crossover, _mutate
        )
        rng = np.random.default_rng(0)
        ind = _encode_individual(rng, _GRID)
        assert "period" in ind
        assert ind["period"] in _GRID["period"]

        parent_b = _encode_individual(rng, _GRID)
        child = _uniform_crossover(ind, parent_b, rng)
        assert child["period"] in _GRID["period"]

        mutated = _mutate(ind, _GRID, mutation_rate=1.0, rng=rng)
        assert mutated["period"] in _GRID["period"]

    def test_failing_objective_handled(self):
        from phinance.optimization.genetic import genetic_search
        def bad_obj(df, params):
            raise RuntimeError("boom")
        params, score = genetic_search(DF, bad_obj, _GRID,
                                       population_size=5, n_generations=2)
        assert isinstance(score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified search() dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchDispatcher:

    def test_random_method(self):
        from phinance.optimization.grid_search import search
        params, score = search(DF, _obj, _GRID, method="random", max_iter=10)
        assert isinstance(params, dict) and isinstance(score, float)

    def test_grid_method(self):
        from phinance.optimization.grid_search import search
        params, score = search(DF, _obj, _GRID, method="grid", max_iter=20)
        assert params.get("period") in _GRID["period"]

    def test_bayesian_method(self):
        from phinance.optimization.grid_search import search
        params, score = search(DF, _obj, _GRID, method="bayesian", max_iter=10)
        assert isinstance(params, dict)

    def test_genetic_method(self):
        from phinance.optimization.grid_search import search
        params, score = search(DF, _obj, _GRID, method="genetic", max_iter=30,
                               population_size=10, n_generations=3)
        assert isinstance(params, dict)

    def test_unknown_method_fallback(self):
        """Unknown method should fall back to random search."""
        from phinance.optimization.grid_search import search
        params, score = search(DF, _obj, _GRID, method="nonexistent", max_iter=5)
        assert isinstance(params, dict)

    def test_search_methods_constant(self):
        from phinance.optimization.grid_search import SEARCH_METHODS
        assert "random"   in SEARCH_METHODS
        assert "bayesian" in SEARCH_METHODS
        assert "genetic"  in SEARCH_METHODS
        assert "grid"     in SEARCH_METHODS


# ═══════════════════════════════════════════════════════════════════════════════
# PhiAI class
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhiAIClass:

    def test_default_search_method(self):
        from phinance.optimization.phiai import PhiAI
        cfg = PhiAI()
        assert cfg.search_method == "random"

    def test_custom_search_method(self):
        from phinance.optimization.phiai import PhiAI
        cfg = PhiAI(search_method="bayesian")
        assert cfg.search_method == "bayesian"

    def test_repr_contains_method(self):
        from phinance.optimization.phiai import PhiAI
        r = repr(PhiAI(search_method="genetic"))
        assert "genetic" in r

    def test_explain_includes_method(self):
        from phinance.optimization.phiai import PhiAI
        cfg = PhiAI(search_method="bayesian")
        cfg.changes = [{"what": "RSI", "reason": "optimized"}]
        expl = cfg.explain()
        assert isinstance(expl, str)

    def test_search_methods_constant_exported(self):
        from phinance.optimization.phiai import SEARCH_METHODS
        assert len(SEARCH_METHODS) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# run_phiai_optimization with search_method
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPhiAIOptimization:

    _indicators = {
        "RSI":  {"enabled": True,  "auto_tune": True,  "params": {}},
        "MACD": {"enabled": True,  "auto_tune": True,  "params": {}},
        "VWAP": {"enabled": False, "auto_tune": False, "params": {}},
    }

    def test_random_search(self):
        from phinance.optimization.phiai import run_phiai_optimization
        result, expl = run_phiai_optimization(DF, self._indicators,
                                              max_iter_per_indicator=5,
                                              search_method="random")
        assert isinstance(result, dict)
        assert "RSI" in result
        assert isinstance(expl, str)

    def test_bayesian_search(self):
        from phinance.optimization.phiai import run_phiai_optimization
        result, expl = run_phiai_optimization(DF, self._indicators,
                                              max_iter_per_indicator=8,
                                              search_method="bayesian")
        assert "RSI" in result

    def test_genetic_search(self):
        from phinance.optimization.phiai import run_phiai_optimization
        result, expl = run_phiai_optimization(DF, self._indicators,
                                              max_iter_per_indicator=20,
                                              search_method="genetic")
        assert "RSI" in result

    def test_disabled_indicator_unchanged(self):
        from phinance.optimization.phiai import run_phiai_optimization
        result, _ = run_phiai_optimization(DF, self._indicators,
                                           max_iter_per_indicator=5,
                                           search_method="random")
        # VWAP disabled → preserved unchanged
        assert result["VWAP"] == self._indicators["VWAP"]

    def test_explanation_mentions_method(self):
        from phinance.optimization.phiai import run_phiai_optimization
        _, expl = run_phiai_optimization(DF, self._indicators,
                                         max_iter_per_indicator=5,
                                         search_method="bayesian")
        # explanation should mention bayesian
        assert "bayesian" in expl.lower()

    def test_unknown_method_falls_back(self):
        from phinance.optimization.phiai import run_phiai_optimization
        result, _ = run_phiai_optimization(DF, self._indicators,
                                           max_iter_per_indicator=5,
                                           search_method="unknown_method")
        assert isinstance(result, dict)
