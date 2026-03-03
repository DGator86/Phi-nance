"""
Auto Learning — Automated backtest learning cycles.

Runs lightweight rolling-window backtests, scores regime predictions,
extracts regime-level lessons, and feeds gradient updates back to the
VariableRegistry so the engine continually improves from its own errors.

Architecture
------------
LearningCycleRunner
  ├── run_cycle(ohlcv_df, ...)          — single learning pass
  ├── run_cycles(ohlcv_df, n, ...)      — multiple rolling passes
  ├── compute_regime_scores()           — per-regime accuracy/edge dict
  ├── extract_lessons()                 — grouped Lesson objects
  └── apply_lessons(registry)           — write gradient updates to VariableRegistry

LessonLog
  └── stores each prediction with outcome and per-regime error

RegimeLessons
  └── aggregate stats per regime: accuracy, avg error, calibration

Usage
-----
  >>> runner = LearningCycleRunner(cfg, registry)
  >>> runner.run_cycles(ohlcv_df, n_cycles=5, window_bars=500)
  >>> scores = runner.compute_regime_scores()
  >>> runner.apply_lessons(registry)
  # VariableRegistry energy weights and projection params updated
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Lesson:
    """One recorded prediction with its realized outcome."""
    timestamp:         Any               # datetime or bar index
    regime:            str               # dominant regime at prediction time
    predicted_signal:  float             # composite_signal ∈ [-1, +1]
    actual_return:     float             # realized next-bar log-return
    error:             float             # predicted direction error magnitude
    correct:           bool              # True if sign(signal) == sign(return)
    regime_probs:      Dict[str, float]  # full probability vector at prediction
    confidence:        float = 0.0       # confidence score at prediction time
    # Options-specific (optional)
    structure:         Optional[str]     = None
    options_pnl:       Optional[float]   = None
    iv_regime:         Optional[str]     = None
    gex_regime:        Optional[str]     = None


@dataclass
class RegimeLessons:
    """Aggregated lesson statistics for a single regime."""
    regime:         str
    n_predictions:  int   = 0
    n_correct:      int   = 0
    total_error:    float = 0.0
    total_return:   float = 0.0
    avg_confidence: float = 0.0
    lessons:        List[Lesson] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(self.n_predictions, 1)

    @property
    def avg_error(self) -> float:
        return self.total_error / max(self.n_predictions, 1)

    @property
    def avg_return(self) -> float:
        return self.total_return / max(self.n_predictions, 1)

    @property
    def edge(self) -> float:
        """Expected return per prediction = mean realized P&L when correct."""
        correct = [l.actual_return for l in self.lessons if l.correct]
        return float(np.mean(correct)) if correct else 0.0

    @property
    def calibration_error(self) -> float:
        """Mean absolute deviation between confidence and accuracy (lower = better)."""
        if not self.lessons:
            return 0.0
        return abs(self.accuracy - self.avg_confidence)

    def summary(self) -> str:
        return (
            f"{self.regime}: n={self.n_predictions} acc={self.accuracy:.1%} "
            f"edge={self.edge:.4f} err={self.avg_error:.4f} calib={self.calibration_error:.3f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# LearningCycleRunner
# ──────────────────────────────────────────────────────────────────────────────

class LearningCycleRunner:
    """
    Automated backtest-based learning engine.

    Runs the RegimeEngine over historical OHLCV windows, collects
    bar-by-bar predictions vs outcomes, scores per regime, and
    applies corrective gradient updates to VariableRegistry.

    Parameters
    ----------
    config   : dict — 'auto_learning' sub-dict from config.yaml
    registry : VariableRegistry instance (optional — needed for apply_lessons)
    """

    def __init__(
        self,
        config:   Dict[str, Any],
        registry: Optional[Any] = None,
    ) -> None:
        self.cfg      = config
        self.registry = registry

        self.min_warmup_bars  = int(config.get('min_warmup_bars', 100))
        self.lesson_lr        = float(config.get('lesson_lr', 0.02))
        self.error_clip       = float(config.get('error_clip', 2.0))
        self.accuracy_target  = float(config.get('accuracy_target', 0.55))
        self.lesson_decay     = float(config.get('lesson_decay', 0.95))

        self._lesson_log:    List[Lesson]                      = []
        self._regime_stats:  Dict[str, RegimeLessons]          = {}
        self._cycle_results: List[Dict[str, Any]]              = []

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        ohlcv_df:     pd.DataFrame,
        window_bars:  int  = 500,
        stride_bars:  int  = 50,
        chain_df:     Optional[pd.DataFrame] = None,
        config:       Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run one learning cycle over ohlcv_df with a rolling window.

        Parameters
        ----------
        ohlcv_df    : DataFrame with columns [open, high, low, close, volume]
        window_bars : number of bars per rolling window
        stride_bars : bars between window starts
        chain_df    : optional options chain for IV regime context
        config      : RegimeEngine config (uses self.cfg['engine_config'] if None)

        Returns
        -------
        dict with: n_lessons, regime_scores, cycle_accuracy, elapsed_s
        """
        from regime_engine.scanner import RegimeEngine

        t0 = time.perf_counter()

        cfg = config or self.cfg.get('engine_config', {})
        engine = RegimeEngine(cfg)

        n_bars = len(ohlcv_df)
        if n_bars < self.min_warmup_bars + 2:
            logger.warning("auto_learning: not enough bars (%d < %d)", n_bars, self.min_warmup_bars)
            return self._empty_cycle_result()

        lessons_this_cycle: List[Lesson] = []
        start_indices = range(0, max(1, n_bars - window_bars - 1), stride_bars)

        for start in start_indices:
            end = min(start + window_bars, n_bars - 1)
            window = ohlcv_df.iloc[start:end].copy()

            if len(window) < self.min_warmup_bars:
                continue

            try:
                result = engine.run(window)
            except Exception as exc:
                logger.debug("auto_learning: engine.run failed at bar %d: %s", start, exc)
                continue

            if result is None:
                continue

            # Realized return for the bar immediately after the window
            next_bar = ohlcv_df.iloc[end] if end < n_bars else None
            if next_bar is None:
                continue

            close_now  = float(window['close'].iloc[-1])
            close_next = float(next_bar['close'])
            if close_now <= 0:
                continue

            actual_return = math.log(close_next / close_now)
            signal        = float(getattr(result, 'composite_signal', 0.0))
            confidence    = float(getattr(result, 'score', 0.0))
            regime_probs  = dict(getattr(result, 'regime_probs', {}))
            dominant      = max(regime_probs, key=lambda k: regime_probs[k]) if regime_probs else 'RANGE'

            predicted_dir = 1 if signal > 0 else -1
            actual_dir    = 1 if actual_return > 0 else -1
            correct       = (predicted_dir == actual_dir) and (abs(signal) > 0.05)
            error         = float(np.clip(abs(signal - actual_return), 0, self.error_clip))

            ts = ohlcv_df.index[end] if hasattr(ohlcv_df.index, '__getitem__') else end

            lesson = Lesson(
                timestamp        = ts,
                regime           = dominant,
                predicted_signal = signal,
                actual_return    = actual_return,
                error            = error,
                correct          = correct,
                regime_probs     = regime_probs,
                confidence       = confidence,
            )
            lessons_this_cycle.append(lesson)
            self._lesson_log.append(lesson)
            self._update_regime_stats(lesson)

        cycle_acc = (
            sum(1 for l in lessons_this_cycle if l.correct) / max(len(lessons_this_cycle), 1)
        )
        elapsed = time.perf_counter() - t0

        cycle_result = {
            'n_lessons':     len(lessons_this_cycle),
            'regime_scores': self.compute_regime_scores(),
            'cycle_accuracy': cycle_acc,
            'elapsed_s':     elapsed,
            'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        self._cycle_results.append(cycle_result)

        logger.info(
            "auto_learning: cycle done | %d lessons | accuracy=%.1f%% | %.1fs",
            len(lessons_this_cycle), cycle_acc * 100, elapsed,
        )
        return cycle_result

    def run_cycles(
        self,
        ohlcv_df:    pd.DataFrame,
        n_cycles:    int = 3,
        window_bars: int = 500,
        stride_bars: int = 50,
        chain_df:    Optional[pd.DataFrame] = None,
        config:      Optional[Dict[str, Any]] = None,
        apply_after: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run n_cycles learning cycles, applying lessons to VariableRegistry
        between each cycle if apply_after=True.

        Returns list of per-cycle result dicts.
        """
        results = []
        for i in range(n_cycles):
            logger.info("auto_learning: starting cycle %d/%d", i + 1, n_cycles)
            r = self.run_cycle(ohlcv_df, window_bars, stride_bars, chain_df, config)
            results.append(r)
            if apply_after and self.registry is not None:
                self.apply_lessons(self.registry)
        return results

    def compute_regime_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-regime accuracy, edge, avg_error, calibration_error.

        Returns
        -------
        dict: {regime_name: {'accuracy': float, 'edge': float, ...}}
        """
        scores: Dict[str, Dict[str, float]] = {}
        for regime, stats in self._regime_stats.items():
            scores[regime] = {
                'accuracy':          stats.accuracy,
                'edge':              stats.edge,
                'avg_error':         stats.avg_error,
                'n_predictions':     stats.n_predictions,
                'calibration_error': stats.calibration_error,
            }
        return scores

    def extract_lessons(self) -> Dict[str, RegimeLessons]:
        """Return all accumulated RegimeLessons grouped by regime."""
        return dict(self._regime_stats)

    def apply_lessons(self, registry: Any) -> None:
        """
        Feed gradient updates from accumulated lessons into VariableRegistry.

        For each regime where accuracy < target, nudge the projection mu
        and tau toward values more consistent with observed returns.
        For well-performing regimes, reward persistence (higher phi).
        """
        scores = self.compute_regime_scores()
        if not scores:
            return

        for regime, s in scores.items():
            n = int(s.get('n_predictions', 0))
            if n < 10:
                continue

            acc   = float(s.get('accuracy', 0.5))
            err   = float(s.get('avg_error', 0.0))
            edge  = float(s.get('edge', 0.0))
            lr    = self.lesson_lr * self.lesson_decay

            # Nudge projection theta_r: if underpredicting returns, push mu up
            if hasattr(registry, 'theta_r'):
                obs_mu = self._regime_stats[regime].avg_return
                theta_r = registry.theta_r
                # VariableRegistry stores theta_r as Dict[regime_name, Dict]
                if isinstance(theta_r, dict) and regime in theta_r:
                    mu_old = theta_r[regime].get("mu", 0.0)
                    theta_r[regime]["mu"] = float(mu_old + lr * (obs_mu - mu_old))

            # Update tau via registry if method exists
            if hasattr(registry, 'update_tau') and err > 0:
                tau_obs = 1.0 / (err + 1e-6)
                try:
                    registry.update_tau(tau_obs)
                except Exception:
                    pass

            # Log the action
            action = "⬆ nudge mu up" if acc < self.accuracy_target else "✓ reinforce"
            logger.debug("apply_lessons: %s [%s] acc=%.1f%% edge=%.4f %s", regime, n, acc * 100, edge, action)

    def get_all_lessons(self) -> List[Lesson]:
        """Return the full flat lesson log."""
        return list(self._lesson_log)

    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """Return list of per-cycle summary dicts."""
        return list(self._cycle_results)

    def underperforming_regimes(self, threshold: Optional[float] = None) -> List[str]:
        """Return regimes with accuracy below threshold (defaults to accuracy_target)."""
        thr = threshold if threshold is not None else self.accuracy_target
        return [
            r for r, s in self.compute_regime_scores().items()
            if s.get('accuracy', 0) < thr and s.get('n_predictions', 0) >= 10
        ]

    def best_regimes(self) -> List[Tuple[str, float]]:
        """Return regimes sorted by edge (descending)."""
        scores = self.compute_regime_scores()
        return sorted(
            [(r, float(s.get('edge', 0))) for r, s in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _update_regime_stats(self, lesson: Lesson) -> None:
        if lesson.regime not in self._regime_stats:
            self._regime_stats[lesson.regime] = RegimeLessons(regime=lesson.regime)
        stats = self._regime_stats[lesson.regime]
        stats.n_predictions  += 1
        stats.n_correct      += int(lesson.correct)
        stats.total_error    += lesson.error
        stats.total_return   += lesson.actual_return
        stats.avg_confidence  = (
            stats.avg_confidence * (stats.n_predictions - 1) / stats.n_predictions
            + lesson.confidence / stats.n_predictions
        )
        stats.lessons.append(lesson)

    @staticmethod
    def _get_regime_idx() -> Dict[str, int]:
        """Map regime name → theta_r row index (must match VariableRegistry.REGIME_ORDER)."""
        from regime_engine.variable_registry import REGIME_ORDER
        return {r: i for i, r in enumerate(REGIME_ORDER)}

    @staticmethod
    def _empty_cycle_result() -> Dict[str, Any]:
        return {
            'n_lessons':      0,
            'regime_scores':  {},
            'cycle_accuracy': 0.0,
            'elapsed_s':      0.0,
            'timestamp':      time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Options lesson integration
    # ──────────────────────────────────────────────────────────────────────

    def record_options_lesson(
        self,
        timestamp:        Any,
        regime:           str,
        structure:        str,
        options_pnl:      float,
        iv_regime:        str,
        gex_regime:       str,
        regime_probs:     Dict[str, float],
        confidence:       float,
        predicted_signal: float = 0.0,
        actual_return:    float = 0.0,
    ) -> None:
        """
        Record an options trade outcome as a lesson.

        Called by OptionsStrategy.on_trading_iteration() after P&L is realized.
        """
        correct = (options_pnl > 0)
        error   = float(np.clip(abs(options_pnl), 0, self.error_clip * 100)) / 100.0

        lesson = Lesson(
            timestamp        = timestamp,
            regime           = regime,
            predicted_signal = predicted_signal,
            actual_return    = actual_return,
            error            = error,
            correct          = correct,
            regime_probs     = regime_probs,
            confidence       = confidence,
            structure        = structure,
            options_pnl      = options_pnl,
            iv_regime        = iv_regime,
            gex_regime       = gex_regime,
        )
        self._lesson_log.append(lesson)
        self._update_regime_stats(lesson)

    def options_structure_scorecard(self) -> Dict[str, Dict[str, Any]]:
        """Return accuracy and avg P&L per options structure type."""
        by_struct: Dict[str, Dict[str, Any]] = {}
        for lesson in self._lesson_log:
            if lesson.structure is None:
                continue
            s = lesson.structure
            if s not in by_struct:
                by_struct[s] = {'n': 0, 'wins': 0, 'total_pnl': 0.0}
            by_struct[s]['n']         += 1
            by_struct[s]['wins']      += int(lesson.correct)
            by_struct[s]['total_pnl'] += float(lesson.options_pnl or 0.0)

        result = {}
        for s, d in by_struct.items():
            n = max(d['n'], 1)
            result[s] = {
                'n':        d['n'],
                'accuracy': d['wins'] / n,
                'avg_pnl':  d['total_pnl'] / n,
            }
        return result
