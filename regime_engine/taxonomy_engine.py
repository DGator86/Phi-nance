"""
Taxonomy Engine — KPCOFGS energy model with sticky logits.

Architecture
------------
For each node n at each bar t:

  E_t(n)  = Σ_j  α_{n,j} · tanh( feature_j )          (energy)
  ℓ_t(n)  = (1 - η_L) · ℓ_{t-1}(n) + η_L · E_t(n)    (sticky logit / EWM)

Implemented as vectorized EWM over the full time axis.
Sibling normalization in log-space is handled in probability_field.py.

Node taxonomy (all nodes, regardless of conditional structure):
  Kingdom  : DIR, NDR, TRN
  Phylum   : LV, NV, HV
  Class    : PT, PX, TE  (DIR children)
             BR, RR, AR  (NDR children)
             SR, RB, FB  (TRN children)
  Order    : AGC, RVP, ABS, EXH
  Family   : ALN, CT, CST
  Genus    : RUN, PBM, FLG, VWM, RRO, SRR
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from .features import FeatureEngine


# ──────────────────────────────────────────────────────────────────────────────
# Node catalogue
# ──────────────────────────────────────────────────────────────────────────────

KINGDOM_NODES  = ["DIR", "NDR", "TRN"]
PHYLUM_NODES   = ["LV", "NV", "HV"]
CLASS_NODES    = ["PT", "PX", "TE", "BR", "RR", "AR", "SR", "RB", "FB"]
ORDER_NODES    = ["AGC", "RVP", "ABS", "EXH"]
FAMILY_NODES   = ["ALN", "CT", "CST"]
GENUS_NODES    = ["RUN", "PBM", "FLG", "VWM", "RRO", "SRR"]

ALL_NODES = (
    KINGDOM_NODES + PHYLUM_NODES + CLASS_NODES +
    ORDER_NODES + FAMILY_NODES + GENUS_NODES
)

# Map level name → node list + smoothing alpha key
LEVEL_META = {
    "kingdom": (KINGDOM_NODES,  "kingdom"),
    "phylum":  (PHYLUM_NODES,   "phylum"),
    "class_":  (CLASS_NODES,    "class_"),
    "order":   (ORDER_NODES,    "order"),
    "family":  (FAMILY_NODES,   "family"),
    "genus":   (GENUS_NODES,    "genus"),
}


class TaxonomyEngine:
    """
    Computes per-bar sticky logits for all taxonomy nodes.

    Parameters
    ----------
    config : dict
        The 'taxonomy' sub-dict from config.yaml.

    Usage
    -----
    >>> engine = TaxonomyEngine(cfg['taxonomy'])
    >>> logits_df = engine.compute_logits(features_df)
    # logits_df columns: all node names, index: same as features_df
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._alpha_mats: Dict[str, np.ndarray] | None = None
        self._feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_logits(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sticky logits for every taxonomy node.

        Parameters
        ----------
        features : pd.DataFrame
            Output of FeatureEngine.compute() — robust-normalized features.

        Returns
        -------
        pd.DataFrame
            Columns = all node names, index = same as features.
            Values are sticky logits ℓ_t(n) (real-valued, not probabilities).
        """
        feat_cols = [c for c in FeatureEngine.FEATURE_COLS if c in features.columns]
        F = features[feat_cols].fillna(0.0).values.astype(np.float64)  # (T, F)
        T = F.shape[0]

        # Bounded transform: tanh applied row-wise
        phi_F = np.tanh(F)  # (T, F) ∈ (-1, 1)

        # Build alpha matrices once (cached)
        alpha_mat, node_order = self._build_alpha_matrix(feat_cols)
        # alpha_mat : (num_nodes, num_features)

        # Raw energies E_t(n) for each node
        E = phi_F @ alpha_mat.T  # (T, num_nodes)

        # Apply EWM smoothing (sticky logit) per level
        logits = np.empty_like(E)
        smoothing = self.cfg["smoothing"]

        node_to_idx = {n: i for i, n in enumerate(node_order)}

        for level, (nodes, smooth_key) in LEVEL_META.items():
            alpha_ewm = smoothing[smooth_key]  # EWM alpha (0 < α ≤ 1)
            indices = [node_to_idx[n] for n in nodes]
            for local_i, node_i in enumerate(indices):
                e_col = E[:, node_i]
                logits[:, node_i] = self._ewm_1d(e_col, alpha_ewm)

        return pd.DataFrame(logits, index=features.index, columns=node_order)

    # ------------------------------------------------------------------
    # Streaming / single-bar update
    # ------------------------------------------------------------------

    def step(
        self,
        feature_row: Dict[str, float],
        prev_logits: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Single-bar sticky logit update for streaming use.

        Parameters
        ----------
        feature_row  : dict of {feature_name: value} (normalized)
        prev_logits  : previous bar's logit dict (None → zero init)

        Returns
        -------
        dict of {node_name: logit_value}
        """
        feat_cols = FeatureEngine.FEATURE_COLS
        F = np.array([feature_row.get(c, 0.0) for c in feat_cols], dtype=np.float64)
        phi_F = np.tanh(F)

        alpha_mat, node_order = self._build_alpha_matrix(feat_cols)
        E = alpha_mat @ phi_F  # (num_nodes,)

        smoothing = self.cfg["smoothing"]
        node_to_idx = {n: i for i, n in enumerate(node_order)}

        prev = prev_logits or {}
        new_logits: Dict[str, float] = {}

        for level, (nodes, smooth_key) in LEVEL_META.items():
            eta = smoothing[smooth_key]
            for node in nodes:
                i = node_to_idx[node]
                prev_l = prev.get(node, 0.0)
                new_logits[node] = (1.0 - eta) * prev_l + eta * float(E[i])

        return new_logits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alpha_matrix(
        self, feat_cols: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Build / return cached (alpha_matrix, node_order)."""
        key = tuple(feat_cols)
        if (
            self._alpha_mats is not None
            and self._feature_cols == feat_cols
        ):
            return self._alpha_mats, self._node_order  # type: ignore[attr-defined]

        energy_cfg = self.cfg["energy"]
        node_order = list(ALL_NODES)
        num_nodes = len(node_order)
        num_feats = len(feat_cols)
        feat_idx = {f: i for i, f in enumerate(feat_cols)}

        alpha = np.zeros((num_nodes, num_feats), dtype=np.float64)

        # Flatten config: level_dict → node_dict → {feature: weight}
        level_map = {
            "kingdom": KINGDOM_NODES,
            "phylum":  PHYLUM_NODES,
            "class_":  CLASS_NODES,
            "order":   ORDER_NODES,
            "family":  FAMILY_NODES,
            "genus":   GENUS_NODES,
        }
        for level_key, nodes in level_map.items():
            level_energy = energy_cfg.get(level_key, {})
            for node in nodes:
                node_weights = level_energy.get(node, {})
                ni = node_order.index(node)
                for feat_name, weight in node_weights.items():
                    fi = feat_idx.get(feat_name)
                    if fi is not None:
                        alpha[ni, fi] = float(weight)

        self._alpha_mats = alpha
        self._node_order = node_order
        self._feature_cols = feat_cols
        return alpha, node_order

    @staticmethod
    def _ewm_1d(x: np.ndarray, alpha: float) -> np.ndarray:
        """Compute exponential weighted moving average of 1-D array.

        y[0] = x[0]
        y[t] = (1 - alpha) * y[t-1] + alpha * x[t]
        """
        out = np.empty_like(x)
        if len(x) == 0:
            return out
        out[0] = x[0]
        one_minus = 1.0 - alpha
        for t in range(1, len(x)):
            out[t] = one_minus * out[t - 1] + alpha * x[t]
        return out
