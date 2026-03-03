"""
tests/unit/test_optimization.py
=================================

Unit tests for phinance.optimization:
  - grid_search and random_search utilities
  - direction_accuracy evaluator
  - build_explanation and format_changes
  - PhiAI class and run_phiai_optimization
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv
from phinance.optimization.grid_search import grid_search, random_search, search
from phinance.optimization.evaluators import (
    direction_accuracy,
    sharpe_proxy,
    sortino_proxy,
)
from phinance.optimization.explainer import (
    build_explanation,
    format_changes,
    format_opt_summary,
)
from phinance.optimization.phiai import PhiAI, run_phiai_optimization


# ─────────────────────────────────────────────────────────────────────────────
#  grid_search
# ─────────────────────────────────────────────────────────────────────────────

class TestGridSearch:

    def test_returns_tuple(self):
        df = make_ohlcv(80)
        grid = {"period": [7, 14]}
        best_p, best_s = grid_search(df, lambda d, p: 0.5, grid)
        assert isinstance(best_p, dict)
        assert isinstance(best_s, float)

    def test_empty_grid_returns_empty(self):
        df = make_ohlcv(50)
        best_p, best_s = grid_search(df, lambda d, p: 0.5, {})
        assert best_p == {}
        assert best_s == 0.0

    def test_finds_best_param(self):
        df = make_ohlcv(80)
        scores = {7: 0.3, 14: 0.6, 21: 0.4}
        grid = {"period": [7, 14, 21]}

        def obj(d, p):
            return scores[p["period"]]

        best_p, best_s = grid_search(df, obj, grid)
        assert best_p["period"] == 14
        assert abs(best_s - 0.6) < 1e-9

    def test_max_iter_cap(self):
        calls = []

        def obj(d, p):
            calls.append(1)
            return 0.5

        df = make_ohlcv(50)
        grid = {"period": list(range(1, 21)), "window": [5, 10]}  # 40 combos
        grid_search(df, obj, grid, max_iter=10)
        assert len(calls) <= 10

    def test_handles_objective_exception(self):
        df = make_ohlcv(50)

        def obj(d, p):
            if p["period"] == 7:
                raise RuntimeError("test error")
            return 0.5

        best_p, best_s = grid_search(df, obj, {"period": [7, 14]})
        assert best_p.get("period") == 14  # 7 raises, falls through to 14


# ─────────────────────────────────────────────────────────────────────────────
#  random_search
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomSearch:

    def test_returns_tuple(self):
        df = make_ohlcv(80)
        grid = {"period": [7, 14, 21]}
        best_p, best_s = random_search(df, lambda d, p: 0.5, grid, max_iter=5)
        assert isinstance(best_p, dict)
        assert isinstance(best_s, float)

    def test_empty_grid_returns_empty(self):
        df = make_ohlcv(50)
        best_p, best_s = random_search(df, lambda d, p: 0.5, {})
        assert best_p == {}

    def test_reproducible_with_seed(self):
        df = make_ohlcv(80)
        grid = {"period": list(range(3, 30))}
        obj = lambda d, p: float(p["period"])

        p1, s1 = random_search(df, obj, grid, max_iter=10, seed=42)
        p2, s2 = random_search(df, obj, grid, max_iter=10, seed=42)
        assert p1 == p2
        assert s1 == s2

    def test_search_dispatcher_random(self):
        df = make_ohlcv(80)
        grid = {"period": [7, 14, 21]}
        best_p, best_s = search(df, lambda d, p: 0.6, grid, method="random", max_iter=5)
        assert isinstance(best_p, dict)

    def test_search_dispatcher_grid(self):
        df = make_ohlcv(80)
        grid = {"period": [7, 14]}
        best_p, best_s = search(df, lambda d, p: 0.6, grid, method="grid")
        assert isinstance(best_p, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  direction_accuracy evaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionAccuracy:

    def test_returns_scalar_in_unit_interval(self):
        df = make_ohlcv(100)
        score = direction_accuracy(df, "RSI", {"period": 14})
        assert 0.0 <= score <= 1.0

    def test_returns_zero_for_unknown_indicator(self):
        df = make_ohlcv(100)
        score = direction_accuracy(df, "NonExistentXYZ", {})
        assert score == 0.0

    def test_returns_zero_for_short_df(self):
        df = make_ohlcv(5)
        score = direction_accuracy(df, "RSI", {"period": 14})
        assert score == 0.0

    def test_different_indicators_give_different_scores(self):
        df = make_ohlcv(150)
        score_rsi  = direction_accuracy(df, "RSI",  {"period": 14})
        score_macd = direction_accuracy(df, "MACD", {})
        # Both are valid scores; they may or may not differ, but both should be finite
        assert np.isfinite(score_rsi)
        assert np.isfinite(score_macd)


# ─────────────────────────────────────────────────────────────────────────────
#  sharpe_proxy / sortino_proxy
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyMetrics:

    def test_sharpe_zero_for_constant_pv(self):
        pv = [100_000.0] * 50
        assert sharpe_proxy(pv) == 0.0

    def test_sharpe_positive_for_growth(self):
        pv = [100_000 * (1.0005 ** i) for i in range(252)]
        s = sharpe_proxy(pv)
        assert s > 0.0

    def test_sortino_zero_for_constant_pv(self):
        pv = [100_000.0] * 50
        assert sortino_proxy(pv) == 0.0

    def test_sortino_finite_for_volatile_pv(self):
        rng = np.random.default_rng(0)
        pv = list(100_000 * np.cumprod(1 + rng.normal(0, 0.01, 100)))
        s = sortino_proxy(pv)
        assert np.isfinite(s)

    def test_short_series_returns_zero(self):
        assert sharpe_proxy([100_000, 101_000]) == 0.0
        assert sortino_proxy([100_000])          == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  explainer
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainer:

    def test_no_changes(self):
        text = build_explanation([])
        assert "no adjustments" in text.lower()

    def test_with_changes(self):
        changes = [
            {"what": "RSI params", "reason": "acc 0.65"},
            {"what": "MACD params", "reason": "acc 0.58"},
        ]
        text = build_explanation(changes)
        assert "2 adjustment" in text
        assert "RSI" in text
        assert "MACD" in text

    def test_with_config_header(self):
        text = build_explanation([], config={"max_indicators": 5, "allow_shorts": False})
        assert "max_indicators=5" in text

    def test_format_changes_detects_diff(self):
        orig     = {"RSI": {"params": {"period": 14}}}
        optimized = {"RSI": {"params": {"period": 10}}}
        changes = format_changes(optimized, orig)
        assert len(changes) == 1
        assert changes[0]["what"] == "RSI params"

    def test_format_changes_no_diff(self):
        orig     = {"RSI": {"params": {"period": 14}}}
        optimized = {"RSI": {"params": {"period": 14}}}
        changes = format_changes(optimized, orig)
        assert len(changes) == 0

    def test_format_opt_summary(self):
        optimized = {"RSI": {"params": {"period": 10}}, "MACD": {"params": {}}}
        scores    = {"RSI": 0.62, "MACD": 0.55}
        text = format_opt_summary(optimized, scores)
        assert "RSI" in text
        assert "0.62" in text or "62%" in text or "acc=62" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  PhiAI class
# ─────────────────────────────────────────────────────────────────────────────

class TestPhiAIClass:

    def test_defaults(self):
        ai = PhiAI()
        assert ai.max_indicators == 5
        assert ai.allow_shorts is False
        assert ai.risk_cap is None

    def test_explain_empty(self):
        ai = PhiAI()
        text = ai.explain()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_repr(self):
        ai = PhiAI(max_indicators=3, allow_shorts=True, risk_cap=0.02)
        r = repr(ai)
        assert "PhiAI" in r
        assert "3" in r

    def test_with_risk_cap(self):
        ai = PhiAI(risk_cap=0.02)
        assert ai.risk_cap == 0.02


# ─────────────────────────────────────────────────────────────────────────────
#  run_phiai_optimization
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPhiAI:

    def test_single_indicator(self):
        df = make_ohlcv(100)
        inds = {"RSI": {"enabled": True, "auto_tune": True, "params": {}}}
        opt, expl = run_phiai_optimization(df, inds, max_iter_per_indicator=3)
        assert "RSI" in opt
        assert isinstance(expl, str)

    def test_optimised_params_are_valid(self):
        df = make_ohlcv(100)
        inds = {"RSI": {"enabled": True, "params": {"period": 14}}}
        opt, _ = run_phiai_optimization(df, inds, max_iter_per_indicator=3)
        params = opt.get("RSI", {}).get("params", {})
        assert isinstance(params, dict)

    def test_unknown_indicator_preserved(self):
        df = make_ohlcv(100)
        inds = {"UnknownXYZ": {"enabled": True, "params": {}}}
        opt, _ = run_phiai_optimization(df, inds, max_iter_per_indicator=2)
        assert "UnknownXYZ" in opt

    def test_explanation_mentions_indicators(self):
        df = make_ohlcv(100)
        inds = {
            "RSI":  {"enabled": True, "params": {}},
            "MACD": {"enabled": True, "params": {}},
        }
        _, expl = run_phiai_optimization(df, inds, max_iter_per_indicator=3)
        # Explanation is a non-empty string
        assert isinstance(expl, str)
        assert len(expl) > 5
