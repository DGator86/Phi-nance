"""
Probability Field — Log-space probability propagation through KPCOFGS hierarchy.

For each bar t the probability of a taxonomy node n is:

  log P(Kingdom k)  = ℓ_t(k) - logsumexp(ℓ_t(siblings_kingdom))
  log P(Phylum v)   = ℓ_t(v) - logsumexp(ℓ_t(siblings_phylum))
  log P(Class c | k) = ℓ_t(c) - logsumexp(ℓ_t(siblings_class(k)))
  log P(Order o)    = ℓ_t(o) - logsumexp(ℓ_t(siblings_order))
  log P(Family f)   = ℓ_t(f) - logsumexp(ℓ_t(siblings_family))
  log P(Genus g)    = ℓ_t(g) - logsumexp(ℓ_t(siblings_genus))

Species log-probability (branch product in log-space):
  log P(s) = log P(k_s) + log P(v_s) + log P(c_s|k_s)
           + log P(o_s) + log P(f_s) + log P(g_s)

Species are re-normalized so they sum to 1.

8 Collapsed regime probabilities are derived using:
  - TREND base → split UP / DN by sigmoid(drift_tscore_30)
  - BREAKOUT base → split UP / DN by sigmoid(drift_tscore_30)
  - LOWVOL / HIGHVOL → phylum-level probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from typing import Dict, Any, List, Tuple

from .taxonomy_engine import (
    TaxonomyEngine,
    KINGDOM_NODES, PHYLUM_NODES, CLASS_NODES,
    ORDER_NODES, FAMILY_NODES, GENUS_NODES,
)
from .species import SPECIES_LIST, CLASS_SIBLINGS, REGIME_BINS


# ──────────────────────────────────────────────────────────────────────────────
# Sibling group definitions for normalization
# ──────────────────────────────────────────────────────────────────────────────

SIBLING_GROUPS: Dict[str, List[str]] = {
    "kingdom": KINGDOM_NODES,
    "phylum":  PHYLUM_NODES,
    "class_DIR": ["PT", "PX", "TE"],
    "class_NDR": ["BR", "RR", "AR"],
    "class_TRN": ["SR", "RB", "FB"],
    "order":   ORDER_NODES,
    "family":  FAMILY_NODES,
    "genus":   GENUS_NODES,
}


def _logsumexp_cols(logits_df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Row-wise logsumexp of selected columns."""
    arr = logits_df[cols].values
    return pd.Series(
        logsumexp(arr, axis=1),
        index=logits_df.index,
    )


class ProbabilityField:
    """
    Converts sticky logits into a full log-probability field.

    Parameters
    ----------
    config : dict  (full engine config, not just one sub-section)

    Usage
    -----
    >>> field = ProbabilityField(cfg)
    >>> log_probs = field.compute(logits_df, features_df)
    # Returns dict with keys:
    #   'nodes'   — DataFrame of per-node log-probs
    #   'species' — DataFrame of per-species log-probs (normalized)
    #   'regimes' — DataFrame of 8-bin regime probs (linear scale, sum=1)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        logits_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Parameters
        ----------
        logits_df   : output of TaxonomyEngine.compute_logits()
        features_df : output of FeatureEngine.compute()

        Returns
        -------
        dict with keys 'nodes', 'species', 'regimes'
        """
        node_lp = self._node_log_probs(logits_df)
        species_lp = self._species_log_probs(node_lp)
        regime_p = self._regime_probs(species_lp, features_df)

        return {
            "nodes":   node_lp,
            "species": species_lp,
            "regimes": regime_p,
        }

    # ------------------------------------------------------------------
    # Node log-probabilities
    # ------------------------------------------------------------------

    def _node_log_probs(self, logits: pd.DataFrame) -> pd.DataFrame:
        """Normalize each sibling group in log-space."""
        lp = pd.DataFrame(index=logits.index)

        # Kingdom
        lse = _logsumexp_cols(logits, KINGDOM_NODES)
        for n in KINGDOM_NODES:
            lp[n] = logits[n] - lse

        # Phylum
        lse = _logsumexp_cols(logits, PHYLUM_NODES)
        for n in PHYLUM_NODES:
            lp[n] = logits[n] - lse

        # Class — conditional on kingdom (3 separate normalizations)
        for kingdom, siblings in [
            ("DIR", ["PT", "PX", "TE"]),
            ("NDR", ["BR", "RR", "AR"]),
            ("TRN", ["SR", "RB", "FB"]),
        ]:
            lse = _logsumexp_cols(logits, siblings)
            for n in siblings:
                # log P(class=n | kingdom) — sibling-only normalization
                lp[f"{n}|{kingdom}"] = logits[n] - lse

        # Order
        lse = _logsumexp_cols(logits, ORDER_NODES)
        for n in ORDER_NODES:
            lp[n] = logits[n] - lse

        # Family
        lse = _logsumexp_cols(logits, FAMILY_NODES)
        for n in FAMILY_NODES:
            lp[n] = logits[n] - lse

        # Genus
        lse = _logsumexp_cols(logits, GENUS_NODES)
        for n in GENUS_NODES:
            lp[n] = logits[n] - lse

        return lp

    # ------------------------------------------------------------------
    # Species log-probabilities
    # ------------------------------------------------------------------

    def _species_log_probs(self, node_lp: pd.DataFrame) -> pd.DataFrame:
        """
        Sum log-probs along each species' branch path.
        Re-normalize so exp(log_probs).sum(axis=1) == 1.
        """
        raw: Dict[str, pd.Series] = {}

        for sp in SPECIES_LIST:
            k = sp.kingdom
            v = sp.phylum
            c = sp.class_
            o = sp.order
            f = sp.family
            g = sp.genus

            log_p = (
                node_lp[k]           # log P(kingdom)
                + node_lp[v]         # log P(phylum)
                + node_lp[f"{c}|{k}"] # log P(class | kingdom)
                + node_lp[o]         # log P(order)
                + node_lp[f]         # log P(family)
                + node_lp[g]         # log P(genus)
            )
            raw[sp.id] = log_p

        raw_df = pd.DataFrame(raw, index=node_lp.index)  # (T, 28)

        # Re-normalize species in log-space
        sp_lse = pd.Series(
            logsumexp(raw_df.values, axis=1),
            index=raw_df.index,
        )
        norm_df = raw_df.subtract(sp_lse, axis=0)
        return norm_df  # log-probabilities, each row sums to 0 in exp-space

    # ------------------------------------------------------------------
    # 8-bin collapsed regime probabilities
    # ------------------------------------------------------------------

    def _regime_probs(
        self,
        species_lp: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Map species probabilities to 8 collapsed regime bins.

        Directional split (UP vs DN) uses sigmoid(drift_tscore_30).
        """
        sp_p = np.exp(species_lp.values)  # (T, 28) linear probs

        # Drift-direction weight: P(up) per bar
        drift = features_df.get(
            "drift_tscore_30",
            pd.Series(0.0, index=features_df.index),
        ).fillna(0.0).values
        p_up = _sigmoid(drift)         # (T,)
        p_dn = 1.0 - p_up

        sp_ids = [sp.id for sp in SPECIES_LIST]
        sp_idx = {sid: i for i, sid in enumerate(sp_ids)}

        # Classify each species
        trend_idx    = [sp_idx[s.id] for s in SPECIES_LIST if s.base_regime == "TREND"]
        range_idx    = [sp_idx[s.id] for s in SPECIES_LIST if s.base_regime == "RANGE"]
        break_idx    = [sp_idx[s.id] for s in SPECIES_LIST if s.base_regime == "BREAKOUT"]
        exhaust_idx  = [sp_idx[s.id] for s in SPECIES_LIST if s.base_regime == "EXHAUST_REV"]
        lowvol_idx   = [sp_idx[s.id] for s in SPECIES_LIST if s.phylum_regime == "LOWVOL"]
        highvol_idx  = [sp_idx[s.id] for s in SPECIES_LIST if s.phylum_regime == "HIGHVOL"]

        def _sum(idx: list) -> np.ndarray:
            if not idx:
                return np.zeros(len(sp_p))
            return sp_p[:, idx].sum(axis=1)

        trend_mass    = _sum(trend_idx)
        range_mass    = _sum(range_idx)
        break_mass    = _sum(break_idx)
        exhaust_mass  = _sum(exhaust_idx)
        lowvol_mass   = _sum(lowvol_idx)
        highvol_mass  = _sum(highvol_idx)

        raw = np.column_stack([
            trend_mass   * p_up,   # TREND_UP
            trend_mass   * p_dn,   # TREND_DN
            range_mass,            # RANGE
            break_mass   * p_up,   # BREAKOUT_UP
            break_mass   * p_dn,   # BREAKOUT_DN
            exhaust_mass,          # EXHAUST_REV
            lowvol_mass,           # LOWVOL
            highvol_mass,          # HIGHVOL
        ])

        # Normalize to sum = 1 per row
        row_sums = raw.sum(axis=1, keepdims=True).clip(min=1e-15)
        norm = raw / row_sums

        return pd.DataFrame(norm, index=species_lp.index, columns=REGIME_BINS)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
