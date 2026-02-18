"""
Expert Registry — Maps indicators to pre-computed signals and validity weights.

The registry:
  1. Holds all active indicator instances.
  2. Resolves validity patterns against species probabilities to compute
     per-indicator soft weights w_j = Σ_{n ∈ V_j} P(n).
  3. Returns a structured dict of {name: signal_series} and {name: weight_series}.

Validity pattern resolution
----------------------------
Patterns follow a hierarchical matching convention:
  'DIR'          → sum P of all DIR species
  'NDR'          → sum P of all NDR species
  'TRN'          → sum P of all TRN species
  'DIR.*.PT'     → sum P of DIR species with class_=PT (wildcards '*' ignored)
  'NDR.*.AR'     → sum P of NDR species with class_=AR
  'TRN.*.FB'     → sum P of TRN species with class_=FB
  'DIR.*.TE'     → sum P of DIR species with class_=TE
  'S08'          → P of species S08 specifically
  'HIGHVOL'      → sum P of all HV phylum species
  'LOWVOL'       → sum P of all LV phylum species
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

from .indicator_library import BaseIndicator, INDICATOR_CLASSES, build_indicator
from .species import SPECIES_LIST


# ──────────────────────────────────────────────────────────────────────────────
# Pattern matcher
# ──────────────────────────────────────────────────────────────────────────────

def _species_ids_for_pattern(pattern: str) -> List[str]:
    """Return list of species IDs matching the given validity pattern."""
    matches: List[str] = []

    for sp in SPECIES_LIST:
        parts = pattern.split(".")

        if pattern.startswith("S") and pattern[1:].isdigit():
            # Exact species ID
            if sp.id == pattern:
                matches.append(sp.id)

        elif pattern == "DIR" and sp.kingdom == "DIR":
            matches.append(sp.id)

        elif pattern == "NDR" and sp.kingdom == "NDR":
            matches.append(sp.id)

        elif pattern == "TRN" and sp.kingdom == "TRN":
            matches.append(sp.id)

        elif pattern == "HIGHVOL" and sp.phylum == "HV":
            matches.append(sp.id)

        elif pattern == "LOWVOL" and sp.phylum == "LV":
            matches.append(sp.id)

        elif len(parts) == 3:
            # kingdom.*.class_  format
            k_pat, _, c_pat = parts
            if (k_pat == "*" or sp.kingdom == k_pat) and \
               (c_pat == "*" or sp.class_  == c_pat):
                matches.append(sp.id)

        elif len(parts) == 2:
            # kingdom.phylum
            k_pat, v_pat = parts
            if (k_pat == "*" or sp.kingdom == k_pat) and \
               (v_pat == "*" or sp.phylum  == v_pat):
                matches.append(sp.id)

    return list(set(matches))


def resolve_validity_weight(
    patterns: List[str],
    species_probs: pd.DataFrame,
) -> pd.Series:
    """
    Compute per-bar soft weight for an indicator.

    w_j = Σ_{n ∈ V_j} P(n)

    Parameters
    ----------
    patterns      : validity patterns from indicator declaration
    species_probs : (T, 28) DataFrame of species probabilities (linear)

    Returns
    -------
    pd.Series of shape (T,), values in [0,1]
    """
    matched_ids: List[str] = []
    for pat in patterns:
        matched_ids.extend(_species_ids_for_pattern(pat))
    matched_ids = list(set(matched_ids))

    if not matched_ids:
        return pd.Series(0.0, index=species_probs.index)

    present = [sid for sid in matched_ids if sid in species_probs.columns]
    if not present:
        return pd.Series(0.0, index=species_probs.index)

    return species_probs[present].sum(axis=1).clip(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Expert Registry
# ──────────────────────────────────────────────────────────────────────────────

class ExpertRegistry:
    """
    Manages a collection of indicators, computes signals and validity weights.

    Parameters
    ----------
    indicator_configs : dict  — from config['indicators']
        {indicator_name: {param_key: value, ...}}

    Usage
    -----
    >>> registry = ExpertRegistry(cfg['indicators'])
    >>> signals, weights = registry.compute(ohlcv_df, species_probs_df)
    """

    def __init__(self, indicator_configs: Dict[str, Any]) -> None:
        self.indicators: Dict[str, BaseIndicator] = {}
        for name, params in indicator_configs.items():
            p = {k: v for k, v in params.items() if k != "type"}
            try:
                self.indicators[name] = build_indicator(name, p)
            except ValueError:
                pass  # Unknown indicator in config — skip gracefully

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ohlcv: pd.DataFrame,
        species_probs: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all indicator signals and corresponding validity weights.

        Parameters
        ----------
        ohlcv         : raw OHLCV DataFrame
        species_probs : (T, 28) DataFrame of species probabilities (linear)
                        — output of exp(probability_field['species'])

        Returns
        -------
        signals : pd.DataFrame  (T, num_indicators) — normalized signals
        weights : pd.DataFrame  (T, num_indicators) — soft validity weights w_j
        """
        sig_dict: Dict[str, pd.Series] = {}
        wt_dict:  Dict[str, pd.Series] = {}

        for name, ind in self.indicators.items():
            try:
                sig = ind.compute(ohlcv).fillna(0.0)
            except Exception:
                sig = pd.Series(0.0, index=ohlcv.index)

            wt = resolve_validity_weight(
                ind.validity_patterns, species_probs
            )
            sig_dict[name] = sig
            wt_dict[name]  = wt

        signals = pd.DataFrame(sig_dict, index=ohlcv.index)
        weights = pd.DataFrame(wt_dict,  index=ohlcv.index)
        return signals, weights

    def indicator_names(self) -> List[str]:
        return list(self.indicators.keys())

    def indicator_type(self, name: str) -> str:
        return self.indicators[name].indicator_type
