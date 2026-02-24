"""
Parameter Tuner — Regime, GEX, and liquidity-conditioned parameter adaptation.

Maintains per-regime parameter banks that override the base config at runtime.
Each bank is updated by lesson feedback from LearningCycleRunner and by
real-time signals from GammaSurface and PolygonL2Client.

Key Dimensions
--------------
  Regime     — 8 MFT regimes; different confidence floors, signal thresholds,
                indicator weights, and position-size multipliers per regime.
  GEX        — gamma_net and gamma_wall_distance adjust confidence thresholds
                and preferred options structures.
  Liquidity  — spread_bps, depth_ratio from L2 feed scale signal damping and
                minimum confidence requirements.

Architecture
------------
ParameterTuner
  ├── tune(regime_probs, gamma_features, l2_signals) → runtime overrides dict
  ├── update_from_lessons(regime_scores)              → adjust regime banks
  ├── update_from_l2(l2_signals)                      → adjust liquidity params
  ├── update_from_gex(gamma_features)                 → adjust GEX-conditioned params
  ├── save(path)                                       → YAML persistence
  └── load(path)                                       → restore from YAML

Runtime Override Dict Keys
--------------------------
  signal_threshold   — minimum |composite_signal| to trigger a trade
  confidence_floor   — minimum confidence score to allow execution
  position_size_mult — scale factor for position size (0.0 = no trade)
  indicator_weights  — per-indicator multipliers (dict[str, float])
  score_scale        — final score scale factor

Usage
-----
  >>> tuner = ParameterTuner(cfg['param_tuner'])
  >>> overrides = tuner.tune(regime_probs, gamma_features, l2_signals)
  >>> # Use overrides in strategy logic:
  >>> threshold = overrides.get('signal_threshold', 0.2)
  >>> floor     = overrides.get('confidence_floor', 0.4)
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Regime order must match variable_registry.REGIME_ORDER
_REGIME_ORDER = [
    'TREND_UP', 'TREND_DN', 'RANGE',
    'BREAKOUT_UP', 'BREAKOUT_DN',
    'EXHAUST_REV', 'LOWVOL', 'HIGHVOL',
]


# ──────────────────────────────────────────────────────────────────────────────
# Default per-regime parameter banks
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_REGIME_BANKS: Dict[str, Dict[str, Any]] = {
    'TREND_UP': {
        'signal_threshold':   0.15,
        'confidence_floor':   0.35,
        'position_size_mult': 1.00,
        'score_scale':        1.00,
        'indicator_weights':  {'rsi': 0.8, 'macd': 1.2, 'momentum': 1.2},
    },
    'TREND_DN': {
        'signal_threshold':   0.15,
        'confidence_floor':   0.35,
        'position_size_mult': 1.00,
        'score_scale':        1.00,
        'indicator_weights':  {'rsi': 0.8, 'macd': 1.2, 'momentum': 1.2},
    },
    'RANGE': {
        'signal_threshold':   0.25,    # higher bar for ranging — more noise
        'confidence_floor':   0.45,
        'position_size_mult': 0.75,
        'score_scale':        0.90,
        'indicator_weights':  {'rsi': 1.3, 'stoch': 1.2, 'macd': 0.7},
    },
    'BREAKOUT_UP': {
        'signal_threshold':   0.10,    # lower bar — capitalize on momentum
        'confidence_floor':   0.30,
        'position_size_mult': 1.25,
        'score_scale':        1.10,
        'indicator_weights':  {'macd': 1.4, 'momentum': 1.4, 'rsi': 0.6},
    },
    'BREAKOUT_DN': {
        'signal_threshold':   0.10,
        'confidence_floor':   0.30,
        'position_size_mult': 1.25,
        'score_scale':        1.10,
        'indicator_weights':  {'macd': 1.4, 'momentum': 1.4, 'rsi': 0.6},
    },
    'EXHAUST_REV': {
        'signal_threshold':   0.20,
        'confidence_floor':   0.50,    # high bar — exhaustion is tricky
        'position_size_mult': 0.60,
        'score_scale':        0.80,
        'indicator_weights':  {'rsi': 1.5, 'stoch': 1.3, 'macd': 0.5},
    },
    'LOWVOL': {
        'signal_threshold':   0.30,    # hard to trade in low vol
        'confidence_floor':   0.50,
        'position_size_mult': 0.50,
        'score_scale':        0.70,
        'indicator_weights':  {},
    },
    'HIGHVOL': {
        'signal_threshold':   0.20,
        'confidence_floor':   0.45,
        'position_size_mult': 0.80,    # risk-managed in high vol
        'score_scale':        0.95,
        'indicator_weights':  {'rsi': 1.2, 'macd': 0.8},
    },
}

# GEX regime adjustments layered on top of regime bank
_GEX_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    'PINNING': {
        # GEX wall — expect range-bound; raise bar for directional trades
        'signal_threshold_add':   0.05,
        'position_size_mult_mul': 0.85,
        'confidence_floor_add':   0.05,
    },
    'AMPLIFY': {
        # Dealers short gamma — price moves amplified; allow lower threshold
        'signal_threshold_add':   -0.03,
        'position_size_mult_mul': 1.15,
        'confidence_floor_add':   -0.03,
    },
    'FLIP': {
        # Near GEX zero-crossing — instability; raise all bars significantly
        'signal_threshold_add':   0.10,
        'position_size_mult_mul': 0.60,
        'confidence_floor_add':   0.15,
    },
    'NEUTRAL': {
        'signal_threshold_add':   0.00,
        'position_size_mult_mul': 1.00,
        'confidence_floor_add':   0.00,
    },
}

# Liquidity-conditioned adjustments from L2 signals
_LIQ_SPREAD_BREAKPOINTS  = [2.0, 5.0, 10.0, 25.0]   # spread_bps → thresholds
_LIQ_DEPTH_BREAKPOINTS   = [0.5, 0.75, 1.0, 1.5]    # depth_ratio → thresholds


# ──────────────────────────────────────────────────────────────────────────────
# ParameterTuner
# ──────────────────────────────────────────────────────────────────────────────

class ParameterTuner:
    """
    Regime × GEX × Liquidity conditioned parameter tuner.

    Combines:
      1. Per-regime parameter bank (tunable via lesson feedback)
      2. Real-time GEX overlay (from GammaSurface)
      3. Real-time liquidity overlay (from PolygonL2Client or REST)

    Parameters
    ----------
    config : dict — 'param_tuner' sub-dict from config.yaml
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.learning_rate    = float(config.get('learning_rate', 0.02))
        self.decay            = float(config.get('decay', 0.995))
        self.max_pos_mult     = float(config.get('max_pos_mult', 2.0))
        self.min_pos_mult     = float(config.get('min_pos_mult', 0.0))
        self.enable_gex       = bool(config.get('enable_gex', True))
        self.enable_liquidity = bool(config.get('enable_liquidity', True))
        self.persist_path     = str(config.get('persist_path', ''))

        # Regime banks: start from defaults, then adapt
        self._banks: Dict[str, Dict[str, Any]] = deepcopy(_DEFAULT_REGIME_BANKS)
        # Merge any user-specified overrides from config
        for regime, user_bank in config.get('regime_banks', {}).items():
            if regime in self._banks:
                self._banks[regime].update(user_bank)

        # Load persisted banks if available
        if self.persist_path and os.path.exists(self.persist_path):
            self.load(self.persist_path)

    # ──────────────────────────────────────────────────────────────────────
    # Primary public API
    # ──────────────────────────────────────────────────────────────────────

    def tune(
        self,
        regime_probs:   Dict[str, float],
        gamma_features: Optional[Dict[str, float]] = None,
        l2_signals:     Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute runtime parameter overrides for the current market state.

        Parameters
        ----------
        regime_probs   : dict from RegimeEngine (8-bin probabilities)
        gamma_features : dict from GammaSurface.compute_features() (optional)
        l2_signals     : dict from PolygonL2Client.get_snapshot() (optional)

        Returns
        -------
        dict with override keys ready for injection into strategy logic.
        """
        dominant = self._dominant(regime_probs)
        base = deepcopy(self._banks.get(dominant, _DEFAULT_REGIME_BANKS.get('RANGE', {})))

        # Blend with second-strongest regime (probability-weighted)
        base = self._blend_regimes(regime_probs, base)

        # Apply GEX overlay
        if self.enable_gex and gamma_features:
            base = self._apply_gex(base, gamma_features)

        # Apply liquidity overlay
        if self.enable_liquidity and l2_signals:
            base = self._apply_liquidity(base, l2_signals)

        # Clamp position size multiplier
        ps = float(base.get('position_size_mult', 1.0))
        base['position_size_mult'] = float(np.clip(ps, self.min_pos_mult, self.max_pos_mult))

        # Always clamp floors
        base['signal_threshold'] = float(np.clip(base.get('signal_threshold', 0.20), 0.05, 0.80))
        base['confidence_floor'] = float(np.clip(base.get('confidence_floor', 0.35), 0.10, 0.90))

        base['dominant_regime'] = dominant
        return base

    # ──────────────────────────────────────────────────────────────────────
    # Lesson-driven updates
    # ──────────────────────────────────────────────────────────────────────

    def update_from_lessons(
        self,
        regime_scores: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Adjust regime parameter banks based on LearningCycleRunner scores.

        If a regime's accuracy is below 50%, tighten its signal_threshold
        and confidence_floor.  If above 60%, allow slightly lower thresholds
        to capture more trades.

        Parameters
        ----------
        regime_scores : output of LearningCycleRunner.compute_regime_scores()
        """
        for regime, scores in regime_scores.items():
            if regime not in self._banks:
                continue
            n   = int(scores.get('n_predictions', 0))
            if n < 10:
                continue

            acc = float(scores.get('accuracy', 0.5))
            err = float(scores.get('avg_error', 0.0))
            lr  = self.learning_rate

            bank = self._banks[regime]

            # Accuracy-driven adjustment to signal_threshold
            delta_thr = lr * (0.55 - acc)   # positive when underperforming
            bank['signal_threshold'] = float(np.clip(
                bank.get('signal_threshold', 0.20) + delta_thr, 0.05, 0.70
            ))

            # Error-driven adjustment to confidence_floor
            delta_floor = lr * err * 0.5
            bank['confidence_floor'] = float(np.clip(
                bank.get('confidence_floor', 0.35) + delta_floor, 0.10, 0.85
            ))

            # Position size: shrink when bad, grow when good
            if acc < 0.45:
                bank['position_size_mult'] = float(np.clip(
                    bank.get('position_size_mult', 1.0) * (1.0 - lr), self.min_pos_mult, self.max_pos_mult
                ))
            elif acc > 0.60:
                bank['position_size_mult'] = float(np.clip(
                    bank.get('position_size_mult', 1.0) * (1.0 + lr * 0.5), self.min_pos_mult, self.max_pos_mult
                ))

            logger.debug(
                "param_tuner.update: %s | acc=%.1f%% | thr→%.3f | floor→%.3f | pos_mult→%.2f",
                regime, acc * 100,
                bank['signal_threshold'], bank['confidence_floor'], bank['position_size_mult'],
            )

        # Apply decay to prevent runaway learning
        self._apply_decay()

        # Persist if path is set
        if self.persist_path:
            self.save(self.persist_path)

    def update_from_gex(self, gamma_features: Dict[str, float]) -> None:
        """
        Real-time GEX signal integration (called each bar).

        Adjusts the RANGE and LOWVOL banks directly when strong pinning
        or amplification is detected, so the next tune() call benefits.
        """
        gex_net = float(gamma_features.get('gamma_net', 0.0))
        flip    = float(gamma_features.get('gex_flip_zone', 0.0))

        if flip >= 0.5:
            # Instability — tighten EXHAUST_REV and RANGE
            for regime in ('EXHAUST_REV', 'RANGE'):
                if regime in self._banks:
                    self._banks[regime]['confidence_floor'] = float(np.clip(
                        self._banks[regime].get('confidence_floor', 0.45) + 0.02, 0.10, 0.90
                    ))
        elif gex_net > 0.5:
            # Strong pinning → RANGE is more reliable
            if 'RANGE' in self._banks:
                self._banks['RANGE']['signal_threshold'] = float(np.clip(
                    self._banks['RANGE'].get('signal_threshold', 0.25) - 0.01, 0.05, 0.70
                ))

    def update_from_l2(self, l2_signals: Dict[str, float]) -> None:
        """
        Real-time L2 liquidity signal integration (called each bar).

        If spread is very wide, raise confidence floors across all regimes.
        """
        spread = float(l2_signals.get('spread_bps', 0.0))
        depth  = float(l2_signals.get('depth_ratio', 1.0))

        if spread > 20:
            # Very illiquid — raise all confidence floors
            for regime in self._banks:
                self._banks[regime]['confidence_floor'] = float(np.clip(
                    self._banks[regime].get('confidence_floor', 0.35) + 0.03, 0.10, 0.90
                ))
        if depth < 0.5:
            # Book very thin — reduce all position sizes
            for regime in self._banks:
                self._banks[regime]['position_size_mult'] = float(np.clip(
                    self._banks[regime].get('position_size_mult', 1.0) * 0.90,
                    self.min_pos_mult, self.max_pos_mult,
                ))

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist all regime banks to YAML."""
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump({'regime_banks': self._banks}, f, default_flow_style=False)
            logger.debug("param_tuner: saved to %s", path)
        except Exception as exc:
            logger.warning("param_tuner.save failed: %s", exc)

    def load(self, path: str) -> None:
        """Load regime banks from YAML, merging with defaults."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            banks = data.get('regime_banks', {})
            for regime, bank in banks.items():
                if regime in self._banks:
                    self._banks[regime].update(bank)
            logger.info("param_tuner: loaded from %s", path)
        except Exception as exc:
            logger.warning("param_tuner.load failed: %s", exc)

    def get_bank(self, regime: str) -> Dict[str, Any]:
        """Return a copy of the parameter bank for a given regime."""
        return deepcopy(self._banks.get(regime, {}))

    def get_all_banks(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of all parameter banks."""
        return deepcopy(self._banks)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dominant(regime_probs: Dict[str, float]) -> str:
        if not regime_probs:
            return 'RANGE'
        return max(regime_probs, key=lambda k: regime_probs[k])

    def _blend_regimes(
        self,
        regime_probs: Dict[str, float],
        dominant_bank: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Probability-weighted blend of the top-2 regime banks.
        Blends scalar parameters (threshold, floor, size multiplier).
        """
        sorted_regimes = sorted(regime_probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_regimes) < 2:
            return dominant_bank

        r1, p1 = sorted_regimes[0]
        r2, p2 = sorted_regimes[1]
        total = p1 + p2
        if total <= 0:
            return dominant_bank

        w1, w2 = p1 / total, p2 / total
        bank2 = self._banks.get(r2, dominant_bank)

        blended = deepcopy(dominant_bank)
        for key in ('signal_threshold', 'confidence_floor', 'position_size_mult', 'score_scale'):
            v1 = float(dominant_bank.get(key, 1.0))
            v2 = float(bank2.get(key, 1.0))
            blended[key] = float(w1 * v1 + w2 * v2)

        return blended

    @staticmethod
    def _apply_gex(bank: Dict[str, Any], gamma_features: Dict[str, float]) -> Dict[str, Any]:
        """Overlay GEX adjustments on top of the blended regime bank."""
        gex_net = float(gamma_features.get('gamma_net', 0.0))
        flip    = float(gamma_features.get('gex_flip_zone', 0.0))
        wall    = float(gamma_features.get('gamma_wall_distance', 0.0))

        if flip >= 0.5:
            gex_regime = 'FLIP'
        elif gex_net >= 0.30:
            gex_regime = 'PINNING'
        elif gex_net <= -0.30:
            gex_regime = 'AMPLIFY'
        else:
            gex_regime = 'NEUTRAL'

        adj = _GEX_ADJUSTMENTS.get(gex_regime, {})

        bank['signal_threshold']   = float(bank.get('signal_threshold', 0.20)   + adj.get('signal_threshold_add', 0.0))
        bank['confidence_floor']   = float(bank.get('confidence_floor', 0.35)   + adj.get('confidence_floor_add', 0.0))
        bank['position_size_mult'] = float(bank.get('position_size_mult', 1.0)  * adj.get('position_size_mult_mul', 1.0))

        # Wall proximity: if spot is very close to a gamma wall, reduce sizing
        if abs(wall) < 0.01:
            bank['position_size_mult'] = float(bank.get('position_size_mult', 1.0) * 0.80)

        bank['_gex_regime'] = gex_regime
        return bank

    @staticmethod
    def _apply_liquidity(bank: Dict[str, Any], l2_signals: Dict[str, float]) -> Dict[str, Any]:
        """Overlay liquidity adjustments from L2 signals."""
        spread     = float(l2_signals.get('spread_bps', 0.0))
        depth_r    = float(l2_signals.get('depth_ratio', 1.0))
        imbalance  = float(l2_signals.get('book_imbalance', 0.5))

        # Spread-based confidence floor bump
        if spread > 25:
            bank['confidence_floor'] = float(bank.get('confidence_floor', 0.35) + 0.15)
        elif spread > 10:
            bank['confidence_floor'] = float(bank.get('confidence_floor', 0.35) + 0.08)
        elif spread > 5:
            bank['confidence_floor'] = float(bank.get('confidence_floor', 0.35) + 0.03)

        # Depth-based position size scaling
        if depth_r < 0.5:
            bank['position_size_mult'] = float(bank.get('position_size_mult', 1.0) * 0.70)
        elif depth_r < 0.75:
            bank['position_size_mult'] = float(bank.get('position_size_mult', 1.0) * 0.85)

        # Order flow imbalance: strong imbalance → lower signal threshold for that direction
        if abs(imbalance - 0.5) > 0.20:
            bank['signal_threshold'] = float(bank.get('signal_threshold', 0.20) * 0.90)

        bank['_liq_spread_bps'] = spread
        bank['_liq_depth_ratio'] = depth_r
        return bank

    def _apply_decay(self) -> None:
        """Decay parameter banks back toward defaults each update cycle."""
        for regime, bank in self._banks.items():
            default = _DEFAULT_REGIME_BANKS.get(regime, {})
            d = self.decay
            for key in ('signal_threshold', 'confidence_floor', 'position_size_mult'):
                current = float(bank.get(key, 1.0))
                dflt    = float(default.get(key, 1.0))
                bank[key] = float(d * current + (1.0 - d) * dflt)
