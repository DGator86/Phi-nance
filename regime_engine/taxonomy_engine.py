"""
Taxonomy Engine — KPCOFGS energy model with sticky logits.

Architecture
------------
For each node n at each bar t:

  gate_i  = f_type(z_i)                                 (per-feature gate)
  E_t(n)  = Σ_j  α_{n,j} · gate_j(feature_j)           (energy)
  ℓ_t(n)  = (1 - η_L) · ℓ_{t-1}(n) + η_L · E_t(n)     (sticky logit / EWM)

Gate types
----------
  tanh      : tanh(γ·z)                         — directional signals
  tanh_rate : tanh(γ·z)  (same but marks Δ-features conceptually)
  tanh_abs  : 1 - tanh(γ·|z|)                  — "near-equilibrium" detection

MSL Condition Matrices
----------------------
The 15×3 MSL→Kingdom matrix and the per-Kingdom-branch MSL→Class matrices are
hardcoded constants derived from the Framework I/O Map specification.  They are
merged *additively* into the alpha matrix alongside the config-file weights, so
config.yaml remains the primary tuning surface.

After raw energy computation, Class energies are scaled by
  sigmoid(logit_parent_kingdom)
so that Class evidence is suppressed when its parent Kingdom is improbable
(Interface 3 context-weighting stage).

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
from typing import Dict, Any, Optional

from .features import FeatureEngine


# ──────────────────────────────────────────────────────────────────────────────
# Node catalogue
# ──────────────────────────────────────────────────────────────────────────────

KINGDOM_NODES = ["DIR", "NDR", "TRN"]
PHYLUM_NODES  = ["LV", "NV", "HV"]
CLASS_NODES   = ["PT", "PX", "TE", "BR", "RR", "AR", "SR", "RB", "FB"]
ORDER_NODES   = ["AGC", "RVP", "ABS", "EXH"]
FAMILY_NODES  = ["ALN", "CT", "CST"]
GENUS_NODES   = ["RUN", "PBM", "FLG", "VWM", "RRO", "SRR"]

ALL_NODES = (
    KINGDOM_NODES + PHYLUM_NODES + CLASS_NODES
    + ORDER_NODES + FAMILY_NODES + GENUS_NODES
)

# Map level name → node list + smoothing alpha key
LEVEL_META = {
    "kingdom": (KINGDOM_NODES, "kingdom"),
    "phylum":  (PHYLUM_NODES,  "phylum"),
    "class_":  (CLASS_NODES,   "class_"),
    "order":   (ORDER_NODES,   "order"),
    "family":  (FAMILY_NODES,  "family"),
    "genus":   (GENUS_NODES,   "genus"),
}

# Kingdom → child Class nodes
CLASS_CHILDREN = {
    "DIR": ["PT", "PX", "TE"],
    "NDR": ["BR", "RR", "AR"],
    "TRN": ["SR", "RB", "FB"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Gate-type registry
# ──────────────────────────────────────────────────────────────────────────────

# Features that need rate-of-change gate (tanh of z directly, but Δ-features)
_RATE_FEATURES = frozenset({"d_mass_dt", "d_lambda", "rv_delta", "er_delta"})

# Features that use the near-equilibrium gate (1 - tanh|z|)
# — active when signal is *close to zero*, not when large
_ABS_FEATURES = frozenset({"mass"})


def _apply_gates(
    F: np.ndarray,
    feat_cols: list[str],
    steepness: dict[str, float],
) -> np.ndarray:
    """
    Apply per-feature gate functions to raw feature matrix.

    Parameters
    ----------
    F          : (T, num_features) raw normalized feature values
    feat_cols  : list of feature names corresponding to columns of F
    steepness  : {feat_name: gamma} override dict; default gamma = 1.0

    Returns
    -------
    phi_F : (T, num_features) gated values
    """
    phi_F = np.empty_like(F)
    default_gamma = steepness.get("default", 1.0)

    for j, col in enumerate(feat_cols):
        gamma = steepness.get(col, default_gamma)
        z = F[:, j]
        if col in _ABS_FEATURES:
            # Near-equilibrium: peak contribution when z ≈ 0 (deep book stable)
            phi_F[:, j] = 1.0 - np.tanh(gamma * np.abs(z))
        else:
            # Directional and rate-of-change: standard tanh
            phi_F[:, j] = np.tanh(gamma * z)

    return phi_F


# ──────────────────────────────────────────────────────────────────────────────
# Interface 2: MSL → Kingdom condition matrix
# ──────────────────────────────────────────────────────────────────────────────
# Rows = MSL features, Cols = Kingdom nodes (DIR, NDR, TRN)
# Values = conditional evidence weight when signal is active; applied after gate.
# Source: Framework I/O Map, Interface 2 condition matrix.

MSL_KINGDOM_MATRIX: dict[str, dict[str, float]] = {
    # market mass (high = deep book)
    "mass": {
        "DIR": -0.6,  # deep book suppresses directional impetus
        "NDR": +0.8,  # depth supports ranging / absorption
        "TRN": -0.3,
    },
    # mass change rate (strongly negative = collapsing book)
    "d_mass_dt": {
        "DIR": +0.2,  # some directional vote when book collapses
        "NDR": -0.4,
        "TRN": +0.9,  # strongest transitional signal
    },
    # dissipation proxy (high = trending field active)
    "dissipation_proxy": {
        "DIR": +0.9,  # high dissipation → strong directional field
        "NDR": -0.7,
        "TRN": +0.3,
    },
    # order-flow imbalance proxy
    "ofi_proxy": {
        "DIR": +0.8,  # directed flow → trending
        "NDR": -0.6,
        "TRN": +0.2,
    },
    # absorption score (volume per price unit)
    "absorption_score": {
        "DIR": -0.5,  # high absorption = range-holding
        "NDR": +0.8,
        "TRN": -0.1,
    },
    # lambda change rate (book thinning speed)
    "d_lambda": {
        "DIR": +0.1,
        "NDR": -0.2,
        "TRN": +0.9,  # rapid lambda change = transitional
    },
    # efficiency ratio (trend persistence)
    "er_60": {
        "DIR": +0.8,
        "NDR": -0.6,
        "TRN": +0.2,
    },
    # ER rising fast → transition signal
    "rv_delta": {
        "DIR": +0.3,
        "NDR": -0.4,
        "TRN": +0.8,
    },
    # realized vol stability feeds phylum more than kingdom; lighter weight here
    "rv_30": {
        "DIR": +0.4,
        "NDR": +0.6,
        "TRN": -0.7,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Interface 3: MSL+Base → Class condition matrices (per Kingdom branch)
# ──────────────────────────────────────────────────────────────────────────────
# Keyed by Kingdom name, then {feature: {class_node: weight}}

MSL_CLASS_MATRICES: dict[str, dict[str, dict[str, float]]] = {
    "DIR": {
        "er_60": {
            "PT": +0.9,   # high ER stable → persistent trend
            "PX": +0.4,
            "TE": -0.5,
        },
        "rv_delta": {
            "PT": -0.4,  # rising vol → TE (exhaustion) not PT
            "PX": +0.9,  # explosive move (PX = price expansion)
            "TE": +0.6,
        },
        "absorption_score": {
            "PT": -0.3,
            "PX": -0.2,
            "TE": +0.8,  # high absorption into trend = exhausting
        },
        "dissipation_proxy": {
            "PT": +0.8,  # trending field → persistent trend
            "PX": +0.3,
            "TE": -0.5,
        },
        "d_mass_dt": {
            "PT": +0.7,  # stable mass → trend persisting
            "PX": +0.3,
            "TE": -0.4,
        },
        "impulse_revert": {
            "PT": -0.4,
            "PX": -0.3,
            "TE": +0.9,  # impulse-revert = exhaustion indicator
        },
    },
    "TRN": {
        "rv_30": {
            "SR": +0.9,  # low RV → squeeze incoming
            "RB": +0.4,
            "FB": -0.2,
        },
        "rv_delta": {
            "SR": -0.2,
            "RB": +0.9,  # rising vol after squeeze → breakout
            "FB": -0.1,
        },
        "impulse_revert": {
            "SR": -0.3,
            "RB": -0.2,
            "FB": +0.9,  # impulse failed → false break
        },
        "d_mass_dt": {
            "SR": +0.6,  # mass collapsing → squeeze pre-release
            "RB": +0.7,
            "FB": -0.3,
        },
        "ofi_proxy": {
            "SR": +0.4,
            "RB": +0.8,  # directional flow confirms breakout
            "FB": -0.2,
        },
        "d_lambda": {
            "SR": +0.7,  # book thinning = squeeze
            "RB": +0.5,
            "FB": -0.1,
        },
    },
    # NDR branch: use config-file weights only (no separate MSL class matrix)
    "NDR": {},
}


# ──────────────────────────────────────────────────────────────────────────────
# TaxonomyEngine
# ──────────────────────────────────────────────────────────────────────────────

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

    Registry integration (optional)
    --------------------------------
    When a VariableRegistry is passed, two behaviours change:

    1. Smoothing constants η_L are read from registry.get_eta(level) instead
       of the fixed config values.  The registry updates η each bar based on
       regime velocity and prediction error.

    2. The learned condition matrix C(i,k,t) from the registry is added to the
       alpha matrix (as C.T), so the engine blends prior knowledge (from
       config.yaml and MSL matrices) with empirically-discovered signal-regime
       relationships.  C starts at zero and adapts over time, so early bars
       behave identically to the non-registry case.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._alpha_mats: np.ndarray | None = None
        self._feature_cols: list[str] | None = None
        self._node_order: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        features: pd.DataFrame,
        registry=None,  # Optional[VariableRegistry]
    ) -> pd.DataFrame:
        """
        Compute sticky logits for every taxonomy node.

        Parameters
        ----------
        features : pd.DataFrame
            Output of FeatureEngine.compute() — robust-normalized features.
        registry : VariableRegistry, optional
            When provided:
            - η_L is read from registry.get_eta(level) instead of config
            - Learned condition matrix C(i,k,t) is blended into alpha_mat
            - η is updated each bar from regime velocity + prediction error

        Returns
        -------
        pd.DataFrame
            Columns = all node names, index = same as features.
            Values are sticky logits ℓ_t(n) (real-valued, not probabilities).
        """
        feat_cols = [c for c in FeatureEngine.FEATURE_COLS if c in features.columns]
        F = features[feat_cols].fillna(0.0).values.astype(np.float64)  # (T, F)
        T = F.shape[0]

        # Per-feature gated transform (Interface 1→2)
        steepness = self.cfg.get("gate_steepness", {})
        phi_F = _apply_gates(F, feat_cols, steepness)  # (T, F) ∈ (-1, 1)

        # Build alpha matrix (base: config + MSL matrices)
        alpha_mat, node_order = self._build_alpha_matrix(feat_cols)
        node_to_idx = {n: i for i, n in enumerate(node_order)}

        # Blend in learned condition matrix from registry if available
        if registry is not None:
            alpha_mat = self._blend_registry_C(alpha_mat, feat_cols, node_order, registry)

        # Raw energies E_t(n) for each node
        E = phi_F @ alpha_mat.T  # (T, num_nodes)

        # Interface 3: context-scale Class energies by parent Kingdom logit
        E = self._apply_class_context_scaling(E, node_to_idx, T)

        # Apply EWM smoothing (sticky logit) per level
        # When registry is present: η is adaptive and must be applied bar-by-bar
        logits = np.empty_like(E)
        smoothing = self.cfg["smoothing"]

        if registry is not None:
            # Sequential application so regime_velocity and η updates are causal
            logits = self._ewm_adaptive(E, node_to_idx, registry)
        else:
            for level, (nodes, smooth_key) in LEVEL_META.items():
                alpha_ewm = smoothing[smooth_key]
                for node in nodes:
                    node_i = node_to_idx[node]
                    logits[:, node_i] = self._ewm_1d(E[:, node_i], alpha_ewm)

        return pd.DataFrame(logits, index=features.index, columns=node_order)

    # ------------------------------------------------------------------
    # Streaming / single-bar update
    # ------------------------------------------------------------------

    def step(
        self,
        feature_row: Dict[str, float],
        prev_logits: Dict[str, float] | None = None,
        registry=None,  # Optional[VariableRegistry]
    ) -> Dict[str, float]:
        """
        Single-bar sticky logit update for streaming use.

        Parameters
        ----------
        feature_row  : dict of {feature_name: value} (normalized)
        prev_logits  : previous bar's logit dict (None → zero init)
        registry     : VariableRegistry — if provided, uses adaptive η_L and
                       blends in the learned condition matrix C(i,k,t).
                       Also updates η_L for each level based on this bar's
                       regime velocity and prediction error.

        Returns
        -------
        dict of {node_name: logit_value}
        """
        feat_cols = FeatureEngine.FEATURE_COLS
        F = np.array(
            [feature_row.get(c, 0.0) for c in feat_cols], dtype=np.float64
        ).reshape(1, -1)
        steepness = self.cfg.get("gate_steepness", {})
        phi_F = _apply_gates(F, feat_cols, steepness)  # (1, F)

        alpha_mat, node_order = self._build_alpha_matrix(feat_cols)

        if registry is not None:
            alpha_mat = self._blend_registry_C(alpha_mat, feat_cols, node_order, registry)

        E = phi_F @ alpha_mat.T  # (1, num_nodes)

        node_to_idx = {n: i for i, n in enumerate(node_order)}
        E = self._apply_class_context_scaling(E, node_to_idx, 1)

        smoothing = self.cfg["smoothing"]
        prev = prev_logits or {}
        new_logits: Dict[str, float] = {}

        for level, (nodes, smooth_key) in LEVEL_META.items():
            if registry is not None:
                eta = registry.get_eta(level)
            else:
                eta = smoothing[smooth_key]

            for node in nodes:
                i = node_to_idx[node]
                prev_l = prev.get(node, 0.0)
                energy = float(E[0, i])
                new_logits[node] = (1.0 - eta) * prev_l + eta * energy

            # Update η via registry: measure regime velocity from logit change
            if registry is not None:
                level_nodes = nodes
                velocity = float(np.mean([
                    abs(new_logits[n] - prev.get(n, 0.0)) for n in level_nodes
                ]))
                # Use logit change as proxy for prediction error
                pred_err = float(np.mean([
                    abs(new_logits[n] - float(E[0, node_to_idx[n]])) for n in level_nodes
                ]))
                registry.update_eta(level, velocity, pred_err)

        return new_logits

    # ------------------------------------------------------------------
    # Registry integration helpers
    # ------------------------------------------------------------------

    def _blend_registry_C(
        self,
        alpha_mat: np.ndarray,
        feat_cols: list[str],
        node_order: list[str],
        registry,  # VariableRegistry
    ) -> np.ndarray:
        """
        Add the registry's learned condition matrix C(i,k,t) to alpha_mat.

        C has shape (n_features, n_nodes); alpha_mat has shape (n_nodes, n_features).
        Adding C.T overlays empirically-learned signal→node weights on top of
        the prior weights from config.yaml and the MSL matrices.

        C starts at zero, so early bars are identical to the non-registry path.
        As the market provides data, C learns the true signal-regime relationships
        and the combined weight vector blends prior + posterior.
        """
        C = registry.get_condition_matrix()  # (n_features, n_nodes) — may be larger
        alpha_out = alpha_mat.copy()

        feat_idx_map = {f: i for i, f in enumerate(feat_cols)}
        node_idx_map = {n: i for i, n in enumerate(node_order)}

        n_f = min(C.shape[0], len(feat_cols))
        n_n = min(C.shape[1], len(node_order))

        for fi in range(n_f):
            for ni in range(n_n):
                if abs(C[fi, ni]) > 1e-9:
                    # C[feature_idx, node_idx] maps to alpha[node_idx, feature_idx]
                    alpha_out[ni, fi] += C[fi, ni]

        return alpha_out

    def _ewm_adaptive(
        self,
        E: np.ndarray,         # (T, num_nodes) — raw energies
        node_to_idx: dict,
        registry,              # VariableRegistry
    ) -> np.ndarray:
        """
        Apply EWM smoothing bar-by-bar using adaptive η_L from the registry.

        Sequential (not vectorized) because η_L changes each bar based on the
        regime velocity observed at that bar — a causal, adaptive process.

        Also updates the registry's η_L after each bar.
        """
        T, num_nodes = E.shape
        logits = np.empty_like(E)
        prev = np.zeros(num_nodes)

        for t in range(T):
            for level, (nodes, _) in LEVEL_META.items():
                eta = registry.get_eta(level)
                level_velocities = []
                for node in nodes:
                    ni = node_to_idx[node]
                    energy = E[t, ni]
                    new_l = (1.0 - eta) * prev[ni] + eta * energy
                    logits[t, ni] = new_l
                    level_velocities.append(abs(new_l - prev[ni]))

                # Regime velocity = mean absolute logit change at this level
                velocity = float(np.mean(level_velocities))
                # Prediction error proxy = smoothing residual
                pred_err = float(np.mean([
                    abs(logits[t, node_to_idx[n]] - E[t, node_to_idx[n]])
                    for n in nodes
                ]))
                registry.update_eta(level, velocity, pred_err)

            prev = logits[t].copy()

        return logits

    # ------------------------------------------------------------------
    # Interface 3 helper: Class context scaling
    # ------------------------------------------------------------------

    def _apply_class_context_scaling(
        self,
        E: np.ndarray,
        node_to_idx: dict[str, int],
        T: int,
    ) -> np.ndarray:
        """
        Scale each Class node's energy by sigmoid(parent_kingdom_logit).

        This implements the Interface 3 blending stage:
          E(class_c) *= sigmoid(E(parent_kingdom_k))

        so that Class evidence is suppressed when its Kingdom parent is
        improbable, falling out naturally from the math without hard gating.

        E is modified in-place and returned.
        """
        def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

        for kingdom, children in CLASS_CHILDREN.items():
            k_idx = node_to_idx.get(kingdom)
            if k_idx is None:
                continue
            # Context weight: how confident are we in this Kingdom?
            context = _sigmoid_vec(E[:, k_idx])  # (T,)
            for cls in children:
                c_idx = node_to_idx.get(cls)
                if c_idx is None:
                    continue
                E[:, c_idx] = E[:, c_idx] * context

        return E

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alpha_matrix(
        self, feat_cols: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Build / return cached (alpha_matrix, node_order)."""
        if (
            self._alpha_mats is not None
            and self._feature_cols == feat_cols
        ):
            return self._alpha_mats, self._node_order  # type: ignore[return-value]

        energy_cfg = self.cfg["energy"]
        msl_scale = float(self.cfg.get("msl_kingdom_scale", 1.0))
        node_order = list(ALL_NODES)
        num_nodes = len(node_order)
        num_feats = len(feat_cols)
        feat_idx = {f: i for i, f in enumerate(feat_cols)}

        alpha = np.zeros((num_nodes, num_feats), dtype=np.float64)

        # ── Layer 1: config-file energy weights ─────────────────────
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
                        alpha[ni, fi] += float(weight)

        # ── Layer 2: MSL Kingdom condition matrix (Interface 2) ──────
        for feat_name, kingdom_votes in MSL_KINGDOM_MATRIX.items():
            fi = feat_idx.get(feat_name)
            if fi is None:
                continue
            for kingdom, weight in kingdom_votes.items():
                ni = node_order.index(kingdom)
                alpha[ni, fi] += msl_scale * weight

        # ── Layer 3: MSL Class condition matrices (Interface 3) ──────
        for kingdom, feat_class_map in MSL_CLASS_MATRICES.items():
            for feat_name, class_votes in feat_class_map.items():
                fi = feat_idx.get(feat_name)
                if fi is None:
                    continue
                for cls_node, weight in class_votes.items():
                    ni = node_order.index(cls_node)
                    alpha[ni, fi] += msl_scale * weight

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
