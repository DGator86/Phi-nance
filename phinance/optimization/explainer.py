"""
phinance.optimization.explainer
================================

Generates human-readable narrative explanations of PhiAI optimisation results.

Usage
-----
    from phinance.optimization.explainer import build_explanation, format_changes

    explanation = build_explanation(changes, config)
    print(explanation)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_explanation(
    changes: List[Dict[str, str]],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a human-readable multi-line explanation of all PhiAI changes.

    Parameters
    ----------
    changes : list of dicts — each has ``"what"`` and ``"reason"`` keys
    config  : dict, optional — PhiAI configuration dict for context header

    Returns
    -------
    str — formatted explanation text
    """
    lines: List[str] = []

    if config:
        max_ind    = config.get("max_indicators", "N/A")
        allow_sh   = config.get("allow_shorts", False)
        risk_cap   = config.get("risk_cap")
        timeframe  = config.get("timeframe", "")
        lines.append(
            f"PhiAI configuration: max_indicators={max_ind}, "
            f"allow_shorts={allow_sh}, risk_cap={risk_cap}"
            + (f", timeframe={timeframe}" if timeframe else "")
        )
        lines.append("")

    if not changes:
        lines.append("PhiAI made no adjustments.")
    else:
        lines.append(f"{len(changes)} adjustment(s) applied:")
        for c in changes:
            what   = c.get("what", "unknown")
            reason = c.get("reason", "")
            lines.append(f"  • {what}: {reason}")

    return "\n".join(lines)


def format_changes(
    optimized: Dict[str, Dict[str, Any]],
    original: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Generate a changes list by comparing optimised vs original indicator configs.

    Parameters
    ----------
    optimized : dict — ``{indicator: {"params": {...}, ...}}``
    original  : dict — same structure before optimisation

    Returns
    -------
    list of ``{"what": str, "reason": str}`` dicts
    """
    changes: List[Dict[str, str]] = []
    for name, new_cfg in optimized.items():
        old_cfg = original.get(name, {})
        new_params = new_cfg.get("params", {})
        old_params = old_cfg.get("params", {})
        if new_params != old_params:
            changes.append({
                "what": f"{name} params",
                "reason": f"Updated {old_params} → {new_params}",
            })
    return changes


def format_opt_summary(
    optimized: Dict[str, Dict[str, Any]],
    scores: Dict[str, float],
) -> str:
    """One-line summary per indicator showing best accuracy score.

    Parameters
    ----------
    optimized : dict
    scores    : dict — ``{indicator_name: accuracy_score}``

    Returns
    -------
    str
    """
    lines = ["PhiAI optimisation results:"]
    for name, score in scores.items():
        cfg = optimized.get(name, {})
        params = cfg.get("params", {})
        lines.append(
            f"  {name}: acc={score:.1%}  params={params}"
        )
    return "\n".join(lines)
