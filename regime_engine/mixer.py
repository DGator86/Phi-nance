"""
Mixer — Composite signal and confidence metric computation.

Composite signal (validity-weighted average of all indicator signals):

  S_t = Σ_j w_{j,t} · z_{j,t}  /  (Σ_j w_{j,t} + ε)

where w_{j,t} = Σ_{n ∈ V_j} P_t(n)   (resolved in ExpertRegistry)

Confidence metrics
------------------

Field confidence — based on entropy of Kingdom/Phylum/Class distributions:

  H(P) = -Σ_k p_k log(p_k)       (Shannon entropy)
  H_max = log(|K|)                (max entropy for |K| categories)
  C_field = 1 - H_norm            where H_norm = H/H_max

  Combined field confidence uses the max entropy across Kingdom, Phylum, Class.

Consensus confidence — agreement of indicator signals (weighted):

  ū   = weighted mean of signals
  σ_u = weighted std of signals
  C_con = |ū| / (|ū| + σ_u + ε)

Liquidity confidence — sigmoid of volume/gap conditions:

  C_liq = sigmoid(vol_zscore / vol_scale) × (1 - sigmoid(|gap| / gap_scale))

Final composite score:

  Score = S_t · C_field · C_con · C_liq
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from typing import Dict, Any, Optional


def _entropy_from_log_probs(log_probs: pd.DataFrame, cols: list) -> pd.Series:
    """Shannon entropy from log-probabilities of a group of nodes."""
    lp = log_probs[cols].values   # (T, K)
    p  = np.exp(np.clip(lp, -500, 0))
    p  = p / p.sum(axis=1, keepdims=True).clip(min=1e-15)
    h  = -(p * np.log(p + 1e-15)).sum(axis=1)
    return pd.Series(h, index=log_probs.index)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Mixer:
    """
    Combines indicator signals and probability field into a composite score.

    Parameters
    ----------
    config : dict — the 'confidence' sub-dict from config.yaml

    Usage
    -----
    >>> mixer = Mixer(cfg['confidence'])
    >>> result = mixer.compute(signals, weights, node_log_probs, features)
    # result is a DataFrame with columns:
    #   composite_signal, c_field, c_consensus, c_liquidity, score
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        signals: pd.DataFrame,
        weights: pd.DataFrame,
        node_log_probs: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        signals        : (T, K) indicator signals (normalized)
        weights        : (T, K) validity weights for each indicator
        node_log_probs : (T, num_nodes) log-probs from ProbabilityField
        features       : (T, F) feature DataFrame (needs volume_zscore, gap_score)

        Returns
        -------
        pd.DataFrame with columns:
          composite_signal, c_field, c_consensus, c_liquidity, score
        """
        eps = self.cfg.get("epsilon", 1e-8)

        s_composite = self._composite_signal(signals, weights, eps)
        c_field     = self._field_confidence(node_log_probs)
        c_consensus = self._consensus_confidence(signals, weights, eps)
        c_liquidity = self._liquidity_confidence(features)

        score = s_composite * c_field * c_consensus * c_liquidity

        return pd.DataFrame(
            {
                "composite_signal": s_composite,
                "c_field":          c_field,
                "c_consensus":      c_consensus,
                "c_liquidity":      c_liquidity,
                "score":            score,
            },
            index=signals.index,
        )

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    def _composite_signal(
        self,
        signals: pd.DataFrame,
        weights: pd.DataFrame,
        eps: float,
    ) -> pd.Series:
        """S_t = Σ_j w_j * z_j / (Σ_j w_j + ε)"""
        w = weights.values.astype(np.float64)
        z = signals.values.astype(np.float64)

        numer = (w * z).sum(axis=1)
        denom = w.sum(axis=1) + eps

        return pd.Series(numer / denom, index=signals.index, name="composite_signal")

    # ------------------------------------------------------------------
    # Field confidence (entropy-based)
    # ------------------------------------------------------------------

    def _field_confidence(self, node_lp: pd.DataFrame) -> pd.Series:
        """
        C_field = 1 − max(H_norm_kingdom, H_norm_phylum, H_norm_class)

        Uses the conditional class probabilities if available.
        """
        from .taxonomy_engine import KINGDOM_NODES, PHYLUM_NODES

        # Kingdom entropy
        k_nodes = [n for n in KINGDOM_NODES if n in node_lp.columns]
        h_k = _entropy_from_log_probs(node_lp, k_nodes)
        h_k_norm = h_k / np.log(max(len(k_nodes), 2))

        # Phylum entropy
        v_nodes = [n for n in PHYLUM_NODES if n in node_lp.columns]
        h_v = _entropy_from_log_probs(node_lp, v_nodes)
        h_v_norm = h_v / np.log(max(len(v_nodes), 2))

        # Class entropy (use first available conditional group)
        class_groups = [
            ["PT|DIR", "PX|DIR", "TE|DIR"],
            ["BR|NDR", "RR|NDR", "AR|NDR"],
            ["SR|TRN", "RB|TRN", "FB|TRN"],
        ]
        h_c_arr = []
        for grp in class_groups:
            present = [c for c in grp if c in node_lp.columns]
            if present:
                h_c_arr.append(_entropy_from_log_probs(node_lp, present)
                               / np.log(max(len(present), 2)))

        # Max normalized entropy across levels
        arrays = [h_k_norm.values, h_v_norm.values]
        for hc in h_c_arr:
            arrays.append(hc.values)

        max_entropy = np.stack(arrays, axis=1).max(axis=1)
        c_field = 1.0 - np.clip(max_entropy, 0.0, 1.0)

        return pd.Series(c_field, index=node_lp.index, name="c_field")

    # ------------------------------------------------------------------
    # Consensus confidence
    # ------------------------------------------------------------------

    def _consensus_confidence(
        self,
        signals: pd.DataFrame,
        weights: pd.DataFrame,
        eps: float,
    ) -> pd.Series:
        """C_con = |ū| / (|ū| + σ_u + ε)"""
        w = weights.values.astype(np.float64)
        z = signals.values.astype(np.float64)

        w_sum = w.sum(axis=1, keepdims=True).clip(min=eps)
        w_norm = w / w_sum

        mean_u = (w_norm * z).sum(axis=1)
        var_u  = (w_norm * (z - mean_u[:, None]) ** 2).sum(axis=1)
        std_u  = np.sqrt(var_u + eps)

        c_con = np.abs(mean_u) / (np.abs(mean_u) + std_u + eps)

        return pd.Series(c_con, index=signals.index, name="c_consensus")

    # ------------------------------------------------------------------
    # Liquidity confidence
    # ------------------------------------------------------------------

    def _liquidity_confidence(self, features: pd.DataFrame) -> pd.Series:
        """
        C_liq = sigmoid(vol_zscore * vol_scale) × sigmoid(-|gap_zscore| * gap_scale)

        Both inputs are robust z-scores (output of FeatureEngine.compute()).
        High volume z-score → more liquidity confidence.
        Large absolute gap z-score → less liquidity confidence (illiquid open).

        vol_scale : multiplier for volume z-score (default 0.5)
        gap_scale : penalty multiplier for gap z-score (default 0.5)
        """
        vol_scale = float(self.cfg.get("liquidity_volume_scale", 0.5))
        gap_scale = float(self.cfg.get("liquidity_gap_scale",    0.5))

        if "volume_zscore" in features.columns:
            vz = features["volume_zscore"].fillna(0.0).values
        else:
            vz = np.zeros(len(features))

        if "gap_score" in features.columns:
            gs = features["gap_score"].fillna(0.0).abs().values
        else:
            gs = np.zeros(len(features))

        c_vol = _sigmoid(vz * vol_scale)
        c_gap = _sigmoid(-gs * gap_scale)

        c_liq = c_vol * c_gap

        return pd.Series(c_liq, index=features.index, name="c_liquidity")

    # ------------------------------------------------------------------
    # Convenience: per-indicator contribution breakdown
    # ------------------------------------------------------------------

    def indicator_contributions(
        self,
        signals: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return per-indicator weighted contribution to composite signal.
        Useful for attribution / AI tuning diagnostics.
        """
        eps = self.cfg.get("epsilon", 1e-8)
        w   = weights.values.astype(np.float64)
        z   = signals.values.astype(np.float64)
        denom = w.sum(axis=1, keepdims=True) + eps

        contrib = (w * z) / denom   # (T, K)
        return pd.DataFrame(contrib, index=signals.index, columns=signals.columns)
