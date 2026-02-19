"""
Scanner — Universe-level regime scanning.

Runs the full engine on a dict of {ticker: ohlcv_df} and returns a ranked
summary DataFrame with regime probabilities, species mode, composite score,
and all confidence metrics.

Compatible with list-of-tickers scanning patterns used in backtesting loops.
"""

from __future__ import annotations

import time
import traceback
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml
import pathlib

from .features import FeatureEngine
from .taxonomy_engine import TaxonomyEngine
from .probability_field import ProbabilityField
from .expert_registry import ExpertRegistry
from .projection_engine import ProjectionEngine
from .mixer import Mixer
from .species import SPECIES_LIST, REGIME_BINS


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerResult:
    ticker:            str
    timestamp:         Any                          # last bar index value
    composite_signal:  float
    score:             float
    c_field:           float
    c_consensus:       float
    c_liquidity:       float
    regime_probs:      Dict[str, float]             # 8-bin
    top_species:       str                          # highest-prob species id
    top_species_prob:  float
    top_species_desc:  str
    projected_signals: Dict[str, float]             # expected next-bar per indicator
    error:             Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Engine (single ticker)
# ──────────────────────────────────────────────────────────────────────────────

class RegimeEngine:
    """
    Full probabilistic regime engine for a single ticker.

    Parameters
    ----------
    config : dict  — full parsed config (from config.yaml)

    Usage
    -----
    >>> engine = RegimeEngine(cfg)
    >>> result = engine.run(ohlcv_df)
    # result is a dict with full time-series DataFrames
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg         = config
        self.features    = FeatureEngine(config["features"])
        self.taxonomy    = TaxonomyEngine(config["taxonomy"])
        self.prob_field  = ProbabilityField(config)
        self.experts     = ExpertRegistry(config.get("indicators", {}))
        self.projection  = ProjectionEngine(config["projection"])
        self.mixer       = Mixer(config["confidence"])

        # Phase 2: optional GammaSurface (lazy import to avoid hard dep)
        gamma_cfg = config.get("gamma", {})
        self._gamma_enabled = bool(gamma_cfg.get("enabled", False))
        self._gamma_surface = None
        if self._gamma_enabled:
            try:
                from .gamma_surface import GammaSurface
                self._gamma_surface = GammaSurface(gamma_cfg)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "GammaSurface init failed — gamma features disabled: %s", exc
                )
                self._gamma_enabled = False

    def run(
        self,
        ohlcv: pd.DataFrame,
        gamma_features: Optional[Dict[str, float]] = None,
        l2_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run full engine on a single OHLCV DataFrame.

        Parameters
        ----------
        ohlcv         : pd.DataFrame with columns open/high/low/close/volume
        gamma_features: optional dict from GammaSurface.compute_features()
                        Keys: gamma_wall_distance, gamma_net,
                              gamma_expiry_days, gex_flip_zone.
                        When provided, values are broadcast as constant
                        columns onto the feature DataFrame.
        l2_features   : optional dict from PolygonL2Client.get_snapshot()
                        Keys: book_imbalance, ofi_true, spread_bps,
                              depth_ratio, depth_trend.
                        Same broadcast treatment as gamma_features.

        Returns
        -------
        dict with keys:
          features, logits, node_log_probs, species_log_probs,
          regime_probs, signals, weights, projections, mix
        """
        # 1. Feature computation
        feat_df = self.features.compute(ohlcv)

        # Inject external feature dicts as constant-valued columns
        for ext_features in [gamma_features, l2_features]:
            if ext_features:
                for col, val in ext_features.items():
                    feat_df[col] = float(val)

        # 2. Taxonomy logits (sticky EWM)
        logits_df = self.taxonomy.compute_logits(feat_df)

        # 3. Probability field (log-space propagation)
        field = self.prob_field.compute(logits_df, feat_df)
        node_lp    = field["nodes"]
        species_lp = field["species"]
        regime_p   = field["regimes"]

        # Linear species probs for validity gating
        species_p  = np.exp(species_lp.clip(upper=0))
        species_p  = species_p.div(species_p.sum(axis=1).clip(lower=1e-15), axis=0)

        # 4. Indicator signals + validity weights
        signals, weights = self.experts.compute(ohlcv, species_p)

        # 5. Projection
        ind_types = {
            n: self.experts.indicator_type(n)
            for n in self.experts.indicator_names()
        }
        proj = self.projection.project(signals, regime_p, ind_types)

        # 6. Composite score
        mix_df = self.mixer.compute(signals, weights, node_lp, feat_df)

        return {
            "features":       feat_df,
            "logits":         logits_df,
            "node_log_probs": node_lp,
            "species_lp":     species_lp,
            "regime_probs":   regime_p,
            "signals":        signals,
            "weights":        weights,
            "projections":    proj,
            "mix":            mix_df,
        }

    def run_latest(self, ohlcv: pd.DataFrame) -> TickerResult:
        """
        Run engine and return a TickerResult representing only the latest bar.
        Efficient for scanner use — full time series computed internally,
        only last row extracted.
        """
        try:
            out = self.run(ohlcv)
        except Exception as e:
            return TickerResult(
                ticker="?", timestamp=None,
                composite_signal=0.0, score=0.0,
                c_field=0.0, c_consensus=0.0, c_liquidity=0.0,
                regime_probs={r: 0.0 for r in REGIME_BINS},
                top_species="?", top_species_prob=0.0, top_species_desc="?",
                projected_signals={},
                error=str(e),
            )

        mix_last     = out["mix"].iloc[-1]
        regime_last  = out["regime_probs"].iloc[-1].to_dict()
        species_last = np.exp(out["species_lp"].iloc[-1])
        top_idx      = int(species_last.values.argmax())
        top_sp       = SPECIES_LIST[top_idx]
        proj_exp     = out["projections"]["expected"].iloc[-1].to_dict()

        return TickerResult(
            ticker            = "",
            timestamp         = ohlcv.index[-1],
            composite_signal  = float(mix_last["composite_signal"]),
            score             = float(mix_last["score"]),
            c_field           = float(mix_last["c_field"]),
            c_consensus       = float(mix_last["c_consensus"]),
            c_liquidity       = float(mix_last["c_liquidity"]),
            regime_probs      = {k: float(v) for k, v in regime_last.items()},
            top_species       = top_sp.id,
            top_species_prob  = float(species_last.iloc[top_idx]),
            top_species_desc  = top_sp.description,
            projected_signals = {k: float(v) for k, v in proj_exp.items()},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Universe Scanner
# ──────────────────────────────────────────────────────────────────────────────

class UniverseScanner:
    """
    Run the regime engine across a universe of tickers and return ranked output.

    Parameters
    ----------
    config_path : str | Path  — path to config.yaml
                  OR
    config : dict             — pre-loaded config dict

    Usage
    -----
    >>> scanner = UniverseScanner(config_path='regime_engine/config.yaml')
    >>> results = scanner.scan(universe)
    # universe: Dict[str, pd.DataFrame]  — {ticker: ohlcv_df}
    """

    def __init__(
        self,
        config_path: Optional[str | pathlib.Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.cfg = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            # Default: look for config.yaml next to this file
            default = pathlib.Path(__file__).parent / "config.yaml"
            with open(default, "r") as f:
                self.cfg = yaml.safe_load(f)

        self.engine      = RegimeEngine(self.cfg)
        self.min_bars    = int(self.cfg.get("scanner", {}).get("min_bars", 300))
        self.top_n       = int(self.cfg.get("scanner", {}).get("top_n", 20))

    def scan(
        self,
        universe: Dict[str, pd.DataFrame],
        sort_by: str = "score",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Scan a universe of tickers.

        Parameters
        ----------
        universe : {ticker: ohlcv_df}
        sort_by  : column to sort output by ('score', 'c_field', 'composite_signal', …)
        ascending: sort direction

        Returns
        -------
        pd.DataFrame  — ranked results, one row per ticker
        """
        rows: List[Dict] = []

        for ticker, ohlcv in universe.items():
            if len(ohlcv) < self.min_bars:
                continue

            result = self.engine.run_latest(ohlcv)
            result.ticker = ticker

            if result.error:
                continue

            row: Dict[str, Any] = {
                "ticker":           ticker,
                "timestamp":        result.timestamp,
                "score":            result.score,
                "composite_signal": result.composite_signal,
                "c_field":          result.c_field,
                "c_consensus":      result.c_consensus,
                "c_liquidity":      result.c_liquidity,
                "top_species":      result.top_species,
                "top_species_prob": result.top_species_prob,
                "top_species_desc": result.top_species_desc,
            }
            # Flatten regime probabilities
            for rbin, prob in result.regime_probs.items():
                row[f"p_{rbin}"] = prob

            # Flatten projected signals
            for ind_name, proj_val in result.projected_signals.items():
                row[f"proj_{ind_name}"] = proj_val

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        return df.head(self.top_n)

    def scan_top_regime(
        self,
        universe: Dict[str, pd.DataFrame],
        regime: str,
        min_prob: float = 0.30,
    ) -> pd.DataFrame:
        """
        Filter universe to tickers where a specific regime has high probability.

        Parameters
        ----------
        regime   : one of REGIME_BINS (e.g. 'TREND_UP', 'RANGE')
        min_prob : minimum probability threshold for the regime

        Returns
        -------
        pd.DataFrame sorted by regime probability descending
        """
        all_results = self.scan(universe, sort_by=f"p_{regime}", ascending=False)
        col = f"p_{regime}"
        if col not in all_results.columns:
            return pd.DataFrame()
        return all_results[all_results[col] >= min_prob].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Optional[str | pathlib.Path] = None) -> Dict[str, Any]:
    """Load and return config dict from YAML file."""
    if path is None:
        path = pathlib.Path(__file__).parent / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
