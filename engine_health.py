"""
Engine Health Check — Verify indicators, regime identifiers, and physics engines.

Runs the full RegimeEngine pipeline on synthetic OHLCV and validates that:
  - FeatureEngine produces expected features
  - TaxonomyEngine produces logits
  - ProbabilityField produces 28 species + 8 regime probabilities
  - ExpertRegistry (indicators) produce signals and weights
  - ProjectionEngine produces expected projections
  - Mixer produces composite score and confidence metrics

Optional components (GammaSurface, Polygon L2, VariableRegistry) are reported
as enabled/disabled/available.

Usage
-----
    from engine_health import run_engine_health_check
    status = run_engine_health_check()
    # status["ok"] == True and status["components"] per-component details
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

# Ensure project root is on path for regime_engine import
if __name__ == "__main__":
    import pathlib
    _root = pathlib.Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from regime_engine import (
    RegimeEngine,
    load_config,
    simulate_ohlcv,
)
from regime_engine.species import REGIME_BINS, SPECIES_LIST


MIN_BARS = 400  # above scanner min_bars (300) for a comfortable margin


def run_engine_health_check(
    n_bars: int = MIN_BARS,
    config_path: str | None = None,
) -> Dict[str, Any]:
    """
    Run the full regime engine on synthetic data and validate every component.

    Returns
    -------
    dict with keys:
      ok          : bool — True if all required components passed
      error       : str | None — top-level exception message if any
      components  : dict — per-component status (ok, message, details)
      optional    : dict — gamma_surface, polygon_l2, variable_registry
    """
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "components": {},
        "optional": {},
    }

    try:
        cfg = load_config(config_path)
    except Exception as e:
        result["error"] = f"Failed to load config: {e}"
        return result

    try:
        engine = RegimeEngine(cfg)
    except Exception as e:
        result["error"] = f"Failed to build RegimeEngine: {e}"
        return result

    ohlcv = simulate_ohlcv(n_bars=n_bars)
    if len(ohlcv) < 300:
        result["error"] = f"Not enough bars: {len(ohlcv)} (need >= 300)"
        return result

    try:
        out = engine.run(ohlcv)
    except Exception as e:
        result["error"] = f"Engine run failed: {e}"
        return result

    comp = result["components"]
    all_ok = True

    # ─── 1. FeatureEngine ─────────────────────────────────────────────────
    feats = out.get("features")
    if feats is None or not hasattr(feats, "shape"):
        comp["feature_engine"] = {"ok": False, "message": "Missing or invalid features DataFrame"}
        all_ok = False
    else:
        n_rows, n_cols = feats.shape
        comp["feature_engine"] = {
            "ok": n_rows >= 300 and n_cols >= 10,
            "message": "Features computed",
            "rows": n_rows,
            "cols": n_cols,
        }
        if not comp["feature_engine"]["ok"]:
            all_ok = False

    # ─── 2. TaxonomyEngine (logits) ───────────────────────────────────────
    logits = out.get("logits")
    if logits is None or not hasattr(logits, "shape"):
        comp["taxonomy_engine"] = {"ok": False, "message": "Missing or invalid logits"}
        all_ok = False
    else:
        comp["taxonomy_engine"] = {
            "ok": len(logits) == len(ohlcv) and logits.shape[1] >= 1,
            "message": "Taxonomy logits computed",
            "nodes": logits.shape[1] if len(logits.shape) > 1 else 0,
        }
        if not comp["taxonomy_engine"]["ok"]:
            all_ok = False

    # ─── 3. ProbabilityField (species + regimes) ───────────────────────────
    species_lp = out.get("species_lp")
    regime_probs = out.get("regime_probs")
    n_species = len(SPECIES_LIST)
    n_regimes = len(REGIME_BINS)

    if species_lp is None or not hasattr(species_lp, "shape"):
        comp["probability_field"] = {"ok": False, "message": "Missing species log-probs"}
        all_ok = False
    elif regime_probs is None or not hasattr(regime_probs, "shape"):
        comp["probability_field"] = {"ok": False, "message": "Missing regime probs"}
        all_ok = False
    else:
        species_ok = species_lp.shape[1] == n_species
        regime_ok = list(regime_probs.columns) == REGIME_BINS
        row_ok = len(regime_probs) == len(ohlcv)
        comp["probability_field"] = {
            "ok": species_ok and regime_ok and row_ok,
            "message": "Species and regime probabilities computed",
            "species_count": n_species,
            "regime_bins": REGIME_BINS,
        }
        if not comp["probability_field"]["ok"]:
            all_ok = False

    # ─── 4. Indicators (ExpertRegistry) ───────────────────────────────────
    signals = out.get("signals")
    weights = out.get("weights")
    ind_names = engine.experts.indicator_names() if hasattr(engine, "experts") else []

    if signals is None or weights is None:
        comp["indicators"] = {
            "ok": False,
            "message": "Missing signals or weights",
            "names": [],
            "count": 0,
        }
        all_ok = False
    else:
        sig_cols = list(signals.columns) if hasattr(signals, "columns") else []
        wt_cols = list(weights.columns) if hasattr(weights, "columns") else []
        match = set(sig_cols) == set(wt_cols) == set(ind_names)
        comp["indicators"] = {
            "ok": match and len(signals) == len(ohlcv),
            "message": "Indicator signals and validity weights computed",
            "names": ind_names,
            "count": len(ind_names),
        }
        if not comp["indicators"]["ok"]:
            all_ok = False

    # ─── 5. ProjectionEngine ────────────────────────────────────────────────
    proj = out.get("projections")
    if proj is None or not isinstance(proj, dict):
        comp["projection_engine"] = {"ok": False, "message": "Missing projections dict"}
        all_ok = False
    else:
        expected = proj.get("expected")
        if expected is None or not hasattr(expected, "columns"):
            comp["projection_engine"] = {"ok": False, "message": "Missing projections['expected']"}
            all_ok = False
        else:
            comp["projection_engine"] = {
                "ok": len(expected) == len(ohlcv) and len(expected.columns) == len(ind_names),
                "message": "AR(1) regime-conditioned projections computed",
                "indicators_projected": list(expected.columns) if hasattr(expected, "columns") else [],
            }
            if not comp["projection_engine"]["ok"]:
                all_ok = False

    # ─── 6. Mixer (composite score + confidence) ────────────────────────────
    mix = out.get("mix")
    required_mix_cols = ["composite_signal", "score", "c_field", "c_consensus", "c_liquidity"]
    if mix is None or not hasattr(mix, "columns"):
        comp["mixer"] = {"ok": False, "message": "Missing mix DataFrame"}
        all_ok = False
    else:
        has_cols = all(c in mix.columns for c in required_mix_cols)
        comp["mixer"] = {
            "ok": has_cols and len(mix) == len(ohlcv),
            "message": "Composite score and confidence metrics computed",
            "columns": required_mix_cols,
        }
        if not comp["mixer"]["ok"]:
            all_ok = False

    # ─── Optional components ───────────────────────────────────────────────
    opt = result["optional"]
    opt["gamma_surface"] = "ok" if getattr(engine, "_gamma_enabled", False) else "disabled"
    poly_cfg = cfg.get("polygon", {}) or {}
    opt["polygon_l2"] = "enabled" if poly_cfg.get("enabled") else "disabled"
    opt["variable_registry"] = "available_not_connected"  # scanner does not wire it by default

    result["ok"] = all_ok
    result["error"] = None
    return result


def format_health_report(status: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable lines for the health report."""
    lines = []
    if status.get("error"):
        lines.append(f"Error: {status['error']}")
        return lines

    lines.append("Regime Engine Health Check")
    lines.append("=" * 40)
    for name, c in status.get("components", {}).items():
        ok_str = "OK" if c.get("ok") else "FAIL"
        lines.append(f"  {name}: [{ok_str}] {c.get('message', '')}")
        for k, v in c.items():
            if k not in ("ok", "message") and v is not None:
                if isinstance(v, list) and len(v) > 5:
                    lines.append(f"      {k}: {len(v)} items")
                else:
                    lines.append(f"      {k}: {v}")

    lines.append("")
    lines.append("Optional")
    lines.append("-" * 40)
    for k, v in status.get("optional", {}).items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Overall: " + ("PASS" if status.get("ok") else "FAIL"))
    return lines


if __name__ == "__main__":
    status = run_engine_health_check()
    for line in format_health_report(status):
        print(line)
    sys.exit(0 if status.get("ok") else 1)
