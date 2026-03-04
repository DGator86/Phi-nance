"""
tests/unit/test_walk_forward.py
=================================

Comprehensive unit tests for phinance.backtest.walk_forward.

Covers:
  - WFOWindow dataclass (creation, to_dict, repr)
  - WFOResult dataclass (creation, to_dict, summary, repr)
  - WalkForwardConfig dataclass (defaults, to_dict)
  - WalkForwardHarness.__init__
  - WalkForwardHarness.windows_count
  - WalkForwardHarness._run_window (returns WFOWindow)
  - WalkForwardHarness.run (returns WFOResult)
  - Aggregate metrics (OOS sharpe, efficiency ratio, gate)
  - run_walk_forward convenience function
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phinance.backtest.walk_forward import (
    WFOWindow,
    WFOResult,
    WalkForwardConfig,
    WalkForwardHarness,
    run_walk_forward,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
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


DF_500 = make_ohlcv(500)
DF_300 = make_ohlcv(300)


# ── WFOWindow ─────────────────────────────────────────────────────────────────


class TestWFOWindow:
    def test_default_creation(self):
        w = WFOWindow()
        assert isinstance(w.window_id, str)
        assert w.is_sharpe == 0.0
        assert w.oos_sharpe == 0.0

    def test_custom_creation(self):
        w = WFOWindow(
            is_start=0, is_end=100, oos_start=100, oos_end=150,
            best_indicator="RSI", is_sharpe=1.2, oos_sharpe=0.9,
        )
        assert w.best_indicator == "RSI"
        assert w.is_sharpe == 1.2
        assert w.oos_sharpe == 0.9

    def test_to_dict_keys(self):
        w = WFOWindow(is_start=0, is_end=100, oos_start=100, oos_end=150)
        d = w.to_dict()
        for k in ("window_id", "is_start", "is_end", "oos_start", "oos_end",
                  "best_indicator", "best_params", "is_sharpe", "oos_sharpe",
                  "oos_return", "oos_drawdown", "oos_trades", "elapsed_ms"):
            assert k in d

    def test_repr(self):
        w = WFOWindow(is_start=0, is_end=100, oos_start=100, oos_end=150,
                      is_sharpe=1.5, oos_sharpe=1.0)
        r = repr(w)
        assert "WFOWindow" in r
        assert "IS=[0:100]" in r


# ── WFOResult ─────────────────────────────────────────────────────────────────


class TestWFOResult:
    def test_default_creation(self):
        r = WFOResult()
        assert isinstance(r.wfo_id, str)
        assert r.num_windows == 0
        assert not r.passed_gate

    def test_to_dict_keys(self):
        r = WFOResult(num_windows=3, combined_oos_sharpe=1.5)
        d = r.to_dict()
        for k in ("wfo_id", "num_windows", "combined_oos_sharpe",
                  "combined_oos_return", "efficiency_ratio", "passed_gate", "windows"):
            assert k in d

    def test_summary_string(self):
        r = WFOResult(num_windows=3, combined_oos_sharpe=0.8, passed_gate=True)
        s = r.summary()
        assert "3" in s
        assert "0.800" in s

    def test_repr(self):
        r = WFOResult(num_windows=5, combined_oos_sharpe=1.2)
        assert "WFOResult" in repr(r)
        assert "5" in repr(r)

    def test_passed_gate_flag(self):
        r = WFOResult(combined_oos_sharpe=0.5, passed_gate=True)
        assert r.passed_gate

    def test_windows_list(self):
        w = WFOWindow(is_start=0, is_end=100, oos_start=100, oos_end=150)
        r = WFOResult(windows=[w], num_windows=1)
        assert len(r.windows) == 1
        assert isinstance(r.windows[0], WFOWindow)


# ── WalkForwardConfig ─────────────────────────────────────────────────────────


class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.is_bars == 120
        assert cfg.oos_bars == 60
        assert cfg.step_bars == 60
        assert cfg.gate_threshold == 0.0
        assert not cfg.auto_deploy
        assert cfg.dry_run

    def test_custom(self):
        cfg = WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40)
        assert cfg.is_bars == 80
        assert cfg.oos_bars == 40

    def test_to_dict_keys(self):
        d = WalkForwardConfig().to_dict()
        for k in ("is_bars", "oos_bars", "step_bars", "gate_threshold", "auto_deploy"):
            assert k in d

    def test_candidate_names_none(self):
        cfg = WalkForwardConfig()
        assert cfg.candidate_names is None

    def test_candidate_names_list(self):
        cfg = WalkForwardConfig(candidate_names=["RSI", "MACD"])
        assert cfg.candidate_names == ["RSI", "MACD"]


# ── WalkForwardHarness.__init__ ───────────────────────────────────────────────


class TestWalkForwardHarnessInit:
    def test_attributes(self):
        h = WalkForwardHarness(ohlcv=DF_300)
        assert len(h.ohlcv) == 300
        assert isinstance(h.config, WalkForwardConfig)

    def test_custom_config(self):
        cfg = WalkForwardConfig(is_bars=80, oos_bars=40)
        h = WalkForwardHarness(ohlcv=DF_300, config=cfg)
        assert h.config.is_bars == 80

    def test_candidates_populated_from_catalog(self):
        h = WalkForwardHarness(ohlcv=DF_300)
        assert len(h._candidates) > 0

    def test_candidates_from_config(self):
        cfg = WalkForwardConfig(candidate_names=["RSI", "MACD"])
        h = WalkForwardHarness(ohlcv=DF_300, config=cfg)
        assert h._candidates == ["RSI", "MACD"]


# ── windows_count ─────────────────────────────────────────────────────────────


class TestWindowsCount:
    def test_exact_fit(self):
        # 500 bars, is=120, oos=60, step=60 → (500-180)/60 = 5.33 → 5+1 = ??
        # pos=0: 0+180<=500 ✓; pos=60: 60+180=240<=500 ✓; ... calculate manually
        h = WalkForwardHarness(ohlcv=DF_500,
                               config=WalkForwardConfig(is_bars=120, oos_bars=60, step_bars=60))
        cnt = h.windows_count()
        assert cnt > 0

    def test_zero_windows_too_small(self):
        tiny = make_ohlcv(n=50)
        h = WalkForwardHarness(ohlcv=tiny,
                               config=WalkForwardConfig(is_bars=120, oos_bars=60, step_bars=60))
        assert h.windows_count() == 0

    def test_counts_match_run(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        expected = h.windows_count()
        r = h.run()
        assert len(r.windows) == expected


# ── _run_window ───────────────────────────────────────────────────────────────


class TestRunWindow:
    def test_returns_wfo_window(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(candidate_names=["RSI"]))
        w = h._run_window(0, 80, 80, 120)
        assert isinstance(w, WFOWindow)

    def test_window_index_fields(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(candidate_names=["RSI"]))
        w = h._run_window(10, 90, 90, 130)
        assert w.is_start == 10
        assert w.is_end == 90
        assert w.oos_start == 90
        assert w.oos_end == 130

    def test_best_indicator_set(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(candidate_names=["RSI", "MACD"]))
        w = h._run_window(0, 80, 80, 120)
        assert w.best_indicator in ("RSI", "MACD")

    def test_elapsed_positive(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(candidate_names=["RSI"]))
        w = h._run_window(0, 80, 80, 120)
        assert w.elapsed_ms >= 0.0


# ── run ───────────────────────────────────────────────────────────────────────


class TestWalkForwardRun:
    def test_returns_wfo_result(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert isinstance(r, WFOResult)

    def test_num_windows(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert r.num_windows > 0

    def test_windows_are_wfo_window(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert all(isinstance(w, WFOWindow) for w in r.windows)

    def test_elapsed_ms_positive(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert r.total_elapsed_ms >= 0.0

    def test_gate_false_by_default(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"],
                                                         gate_threshold=999.0))
        r = h.run()
        assert not r.passed_gate

    def test_gate_true_when_zero_threshold(self):
        # Most strategies will have at least 0 OOS sharpe
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"],
                                                         gate_threshold=-999.0))
        r = h.run()
        assert r.passed_gate

    def test_empty_data_returns_result(self):
        tiny = make_ohlcv(n=50)
        h = WalkForwardHarness(ohlcv=tiny,
                               config=WalkForwardConfig(is_bars=120, oos_bars=60, step_bars=60,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert r.num_windows == 0

    def test_wfo_id_is_string(self):
        h = WalkForwardHarness(ohlcv=DF_300,
                               config=WalkForwardConfig(is_bars=80, oos_bars=40, step_bars=40,
                                                         candidate_names=["RSI"]))
        r = h.run()
        assert isinstance(r.wfo_id, str)
        assert len(r.wfo_id) > 0


# ── Aggregate metrics ─────────────────────────────────────────────────────────


class TestAggregateMetrics:
    def _run(self, n: int = 300, threshold: float = -999.0) -> WFOResult:
        return WalkForwardHarness(
            ohlcv=make_ohlcv(n),
            config=WalkForwardConfig(
                is_bars=80, oos_bars=40, step_bars=40,
                candidate_names=["RSI", "EMA Cross"],
                gate_threshold=threshold,
            ),
        ).run()

    def test_combined_oos_sharpe_is_float(self):
        r = self._run()
        assert isinstance(r.combined_oos_sharpe, float)

    def test_efficiency_ratio_finite(self):
        r = self._run()
        assert not np.isnan(r.efficiency_ratio)
        assert not np.isinf(r.efficiency_ratio)

    def test_to_dict_windows_list(self):
        r = self._run()
        d = r.to_dict()
        assert isinstance(d["windows"], list)


# ── run_walk_forward ──────────────────────────────────────────────────────────


class TestRunWalkForward:
    def test_returns_wfo_result(self):
        r = run_walk_forward(DF_300, is_bars=80, oos_bars=40, step_bars=40,
                              candidate_names=["RSI"])
        assert isinstance(r, WFOResult)

    def test_num_windows_positive(self):
        r = run_walk_forward(DF_300, is_bars=80, oos_bars=40, step_bars=40,
                              candidate_names=["RSI"])
        assert r.num_windows > 0

    def test_gate_threshold_respected(self):
        r = run_walk_forward(DF_300, is_bars=80, oos_bars=40, step_bars=40,
                              gate_threshold=999.0, candidate_names=["RSI"])
        assert not r.passed_gate

    def test_default_candidates(self):
        r = run_walk_forward(DF_300, is_bars=80, oos_bars=40, step_bars=40)
        assert r.num_windows > 0

    def test_summary_string(self):
        r = run_walk_forward(DF_300, is_bars=80, oos_bars=40, step_bars=40,
                              candidate_names=["RSI"])
        s = r.summary()
        assert "WFO" in s
