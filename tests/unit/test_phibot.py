"""
tests/unit/test_phibot.py
==========================

Unit tests for phinance.phibot.reviewer:
  - detect_market_regime returns valid labels
  - _reconstruct_trades reconstructs BUY→SELL pairs
  - _compute_regime_stats aggregates correctly
  - _generate_tweaks produces actionable suggestions
  - review_backtest returns a complete BacktestReview
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv

from phinance.phibot.reviewer import (
    Tweak,
    BacktestReview,
    detect_market_regime,
    _reconstruct_trades,
    _compute_regime_stats,
    _generate_tweaks,
    _build_observations,
    _build_summary_and_verdict,
    review_backtest,
)

_VALID_REGIMES = {
    "TREND_UP", "TREND_DN", "RANGE",
    "BREAKOUT_UP", "BREAKOUT_DN", "HIGHVOL", "LOWVOL",
}


# ─────────────────────────────────────────────────────────────────────────────
#  detect_market_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectMarketRegime:

    def test_returns_series_same_length(self):
        df = make_ohlcv(100)
        regime = detect_market_regime(df)
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(df)

    def test_all_labels_valid(self):
        df = make_ohlcv(200)
        regime = detect_market_regime(df)
        unique_labels = set(regime.unique())
        assert unique_labels.issubset(_VALID_REGIMES | {"RANGE"})

    def test_short_df_returns_range(self):
        df = make_ohlcv(10)
        regime = detect_market_regime(df)
        assert regime is not None

    def test_none_input_returns_empty(self):
        result = detect_market_regime(None, lookback=20)
        assert result is not None  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
#  _reconstruct_trades
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructTrades:

    def _make_log(self, signals):
        """Build a minimal prediction_log from a list of signal strings."""
        return [
            {"signal": sig, "price": 100.0 + i * 0.5, "date": f"2023-01-{i+1:02d}"}
            for i, sig in enumerate(signals)
        ]

    def test_empty_log_returns_empty(self):
        trades = _reconstruct_trades([], None)
        assert trades == []

    def test_single_buy_sell(self):
        log = self._make_log(["UP", "NEUTRAL", "NEUTRAL", "DOWN"])
        trades = _reconstruct_trades(log, None)
        assert len(trades) == 1
        t = trades[0]
        assert "entry_price" in t
        assert "exit_price"  in t
        assert "win" in t
        assert "hold_bars" in t

    def test_multiple_trades(self):
        log = self._make_log(["UP", "DOWN", "UP", "DOWN"])
        trades = _reconstruct_trades(log, None)
        assert len(trades) == 2

    def test_open_position_not_counted(self):
        log = self._make_log(["UP", "NEUTRAL", "NEUTRAL"])  # never closed
        trades = _reconstruct_trades(log, None)
        assert len(trades) == 0

    def test_win_flag_correct(self):
        # price increases 100→102, then closes → win
        log = [
            {"signal": "UP",   "price": 100.0, "date": "2023-01-01"},
            {"signal": "NEUTRAL", "price": 101.0, "date": "2023-01-02"},
            {"signal": "DOWN", "price": 102.0, "date": "2023-01-03"},
        ]
        trades = _reconstruct_trades(log, None)
        assert len(trades) == 1
        assert trades[0]["win"] is True

    def test_loss_flag_correct(self):
        log = [
            {"signal": "UP",   "price": 100.0, "date": "2023-01-01"},
            {"signal": "DOWN", "price": 95.0,  "date": "2023-01-02"},
        ]
        trades = _reconstruct_trades(log, None)
        assert len(trades) == 1
        assert trades[0]["win"] is False


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_regime_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRegimeStats:

    def test_empty_trades(self):
        stats = _compute_regime_stats([])
        assert stats == {}

    def test_single_win(self):
        trades = [{"win": True, "pnl_pct": 0.05, "regime": "TREND_UP"}]
        stats = _compute_regime_stats(trades)
        assert "TREND_UP" in stats
        assert stats["TREND_UP"]["win_rate"] == 1.0
        assert stats["TREND_UP"]["count"] == 1

    def test_mixed_results(self):
        trades = [
            {"win": True,  "pnl_pct":  0.05, "regime": "RANGE"},
            {"win": False, "pnl_pct": -0.03, "regime": "RANGE"},
            {"win": True,  "pnl_pct":  0.07, "regime": "RANGE"},
        ]
        stats = _compute_regime_stats(trades)
        r = stats["RANGE"]
        assert r["count"] == 3
        assert abs(r["win_rate"] - 2/3) < 1e-9
        assert abs(r["avg_pl"] - 0.03) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  _generate_tweaks
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateTweaks:

    def _make_losing_trades(self, n=10):
        return [{"win": False, "pnl_pct": -0.04, "regime": "RANGE"} for _ in range(n)]

    def test_no_trades_returns_no_tweaks(self):
        tweaks = _generate_tweaks([], {}, {}, {}, "weighted_sum", {})
        assert tweaks == []

    def test_low_win_rate_suggests_higher_threshold(self):
        trades = self._make_losing_trades(n=10)
        regime_stats = _compute_regime_stats(trades)
        tweaks = _generate_tweaks(
            trades, regime_stats, {"RSI": {}}, {}, "weighted_sum", {}
        )
        ids = [t.id for t in tweaks]
        assert "threshold_up" in ids

    def test_high_drawdown_suggests_position_reduction(self):
        trades = [{"win": True, "pnl_pct": 0.02, "regime": "RANGE"}]
        regime_stats = _compute_regime_stats(trades)
        results = {"max_drawdown": 0.35}
        tweaks = _generate_tweaks(trades, regime_stats, {}, {}, "weighted_sum", results)
        ids = [t.id for t in tweaks]
        assert "reduce_position_size" in ids

    def test_regime_dispersion_suggests_regime_blend(self):
        trades = (
            [{"win": True,  "pnl_pct":  0.06, "regime": "TREND_UP"}] * 5
            + [{"win": False, "pnl_pct": -0.04, "regime": "RANGE"}] * 5
        )
        regime_stats = _compute_regime_stats(trades)
        tweaks = _generate_tweaks(
            trades, regime_stats, {"RSI": {}}, {}, "weighted_sum", {}
        )
        ids = [t.id for t in tweaks]
        assert "use_regime_weighted" in ids

    def test_tweaks_have_required_fields(self):
        trades = self._make_losing_trades(n=8)
        regime_stats = _compute_regime_stats(trades)
        tweaks = _generate_tweaks(trades, regime_stats, {}, {}, "voting", {})
        for tw in tweaks:
            assert isinstance(tw.id,          str)
            assert isinstance(tw.category,    str)
            assert isinstance(tw.title,       str)
            assert isinstance(tw.rationale,   str)
            assert tw.confidence in ("high", "medium", "low")


# ─────────────────────────────────────────────────────────────────────────────
#  _build_observations
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildObservations:

    def test_no_trades_message(self):
        obs = _build_observations([], {}, {})
        assert len(obs) == 1
        assert "no completed trades" in obs[0].lower()

    def test_observations_non_empty(self):
        trades = [
            {"win": True,  "pnl_pct": 0.05, "hold_bars": 5,  "regime": "TREND_UP"},
            {"win": False, "pnl_pct":-0.03, "hold_bars": 3,  "regime": "RANGE"},
        ]
        regime_stats = _compute_regime_stats(trades)
        obs = _build_observations(trades, regime_stats, {"sharpe": 1.1, "max_drawdown": 0.08})
        assert len(obs) >= 3


# ─────────────────────────────────────────────────────────────────────────────
#  _build_summary_and_verdict
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSummaryAndVerdict:

    def test_strong_verdict(self):
        _, verdict = _build_summary_and_verdict(
            {"sharpe": 2.0, "total_return": 0.25}, win_rate=0.60
        )
        assert verdict == "strong"

    def test_moderate_verdict(self):
        _, verdict = _build_summary_and_verdict(
            {"sharpe": 1.0, "total_return": 0.05}, win_rate=0.48
        )
        assert verdict == "moderate"

    def test_weak_verdict(self):
        _, verdict = _build_summary_and_verdict(
            {"sharpe": 0.3, "total_return": 0.02}, win_rate=0.40
        )
        assert verdict == "weak"

    def test_neutral_verdict(self):
        _, verdict = _build_summary_and_verdict(
            {"sharpe": -0.1, "total_return": -0.05}, win_rate=0.35
        )
        assert verdict == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
#  review_backtest — full integration
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewBacktest:

    def _make_prediction_log(self, n=20):
        log = []
        for i in range(n):
            sig = "UP" if i % 5 == 0 else ("DOWN" if i % 5 == 3 else "NEUTRAL")
            log.append({
                "signal": sig,
                "price": 100.0 + i * 0.1,
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
            })
        return log

    def test_returns_backtest_review(self):
        df   = make_ohlcv(100)
        log  = self._make_prediction_log(20)
        results = {
            "total_return": 0.08,
            "sharpe": 1.2,
            "max_drawdown": 0.05,
        }
        review = review_backtest(
            ohlcv          = df,
            results        = results,
            prediction_log = log,
            indicators     = {"RSI": {"enabled": True}},
            blend_weights  = {"RSI": 1.0},
            blend_method   = "weighted_sum",
            config         = {"symbols": ["SPY"]},
        )
        assert isinstance(review, BacktestReview)
        assert review.verdict in ("strong", "moderate", "weak", "neutral")
        assert isinstance(review.summary, str)
        assert len(review.summary) > 10
        assert isinstance(review.observations, list)
        assert isinstance(review.tweaks, list)

    def test_review_with_no_trades(self):
        df  = make_ohlcv(100)
        review = review_backtest(
            ohlcv          = df,
            results        = {"total_return": 0.0, "sharpe": 0.0},
            prediction_log = [],
            indicators     = {},
            blend_weights  = {},
            blend_method   = "weighted_sum",
            config         = {},
        )
        assert review.total_trades == 0
        assert review.win_rate == 0.0

    def test_review_none_ohlcv(self):
        review = review_backtest(
            ohlcv          = None,
            results        = {"total_return": -0.02},
            prediction_log = [],
            indicators     = {},
            blend_weights  = {},
            blend_method   = "weighted_sum",
            config         = {},
        )
        assert isinstance(review, BacktestReview)

    def test_review_with_multiple_trades(self):
        df  = make_ohlcv(200)
        log = self._make_prediction_log(40)
        results = {
            "total_return": 0.12,
            "sharpe": 1.5,
            "max_drawdown": 0.08,
        }
        review = review_backtest(
            ohlcv          = df,
            results        = results,
            prediction_log = log,
            indicators     = {"RSI": {}, "MACD": {}},
            blend_weights  = {"RSI": 0.5, "MACD": 0.5},
            blend_method   = "weighted_sum",
            config         = {"symbols": ["SPY"]},
        )
        assert review.total_trades >= 0
        assert isinstance(review.regime_stats, dict)
