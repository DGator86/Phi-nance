"""
Options Post-Trade Reviewer — Regime & IV & GEX aware analysis.
================================================================
Analyses a completed options engine backtest and generates actionable
insights with specific tweaks.  Mirrors phi/phibot/reviewer.py but
is purpose-built for the options pipeline (structures, Greeks, IV
regimes, GEX conditions).

Usage
-----
    >>> from phi.options.options_reviewer import review_options_backtest
    >>> review = review_options_backtest(
    ...     trade_log=results["trades"],
    ...     metrics=results,
    ...     ohlcv=ohlcv,
    ... )
    >>> print(review.summary)
    >>> for tweak in review.tweaks:
    ...     print(tweak.title, tweak.rationale)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsTweak:
    """A single actionable suggestion."""
    id:              str
    category:        str      # confidence_gate | position_size | iv_premium |
                              # strategy_mix | exit_rules | dte_selection
    title:           str
    rationale:       str
    current_value:   Any
    suggested_value: Any
    param_key:       str      # session-state key for one-click adoption
    confidence:      str = "medium"  # high | medium | low


@dataclass
class OptionsBacktestReview:
    """Full options backtest review output."""
    summary:            str
    verdict:            str            # strong | moderate | weak | neutral
    total_trades:       int
    win_rate:           float
    avg_hold_days:      float
    avg_pnl_pct:        float
    structure_stats:    Dict[str, Dict]   # structure → {count, wins, avg_pnl}
    regime_stats:       Dict[str, Dict]   # regime → {count, wins, avg_pnl}
    vol_regime_stats:   Dict[str, Dict]   # vol_regime → {count, wins, avg_pnl}
    gex_regime_stats:   Dict[str, Dict]   # gex_regime → {count, wins, avg_pnl}
    observations:       List[str]
    tweaks:             List[OptionsTweak]
    exit_reason_stats:  Dict[str, int]    # exit_reason → count


# ──────────────────────────────────────────────────────────────────────────────
# Structure-regime affinity knowledge base
# ──────────────────────────────────────────────────────────────────────────────

_STRUCTURE_REGIME_AFFINITY: Dict[str, Dict[str, float]] = {
    "long_call":        {"TREND_UP": 1.5, "BREAKOUT_UP": 1.3, "RANGE": 0.4, "TREND_DN": 0.3},
    "long_put":         {"TREND_DN": 1.5, "BREAKOUT_DN": 1.3, "RANGE": 0.4, "TREND_UP": 0.3},
    "bull_call_spread":  {"TREND_UP": 1.4, "BREAKOUT_UP": 1.1, "RANGE": 0.6},
    "bear_put_spread":   {"TREND_DN": 1.4, "BREAKOUT_DN": 1.1, "RANGE": 0.6},
    "iron_condor":       {"RANGE": 1.5, "LOWVOL": 1.3, "TREND_UP": 0.4, "TREND_DN": 0.4},
    "iron_butterfly":    {"RANGE": 1.4, "LOWVOL": 1.2},
    "long_straddle":     {"BREAKOUT_UP": 1.4, "BREAKOUT_DN": 1.4, "HIGHVOL": 1.2, "RANGE": 0.3},
    "long_strangle":     {"BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3, "HIGHVOL": 1.1, "RANGE": 0.3},
    "covered_call":      {"RANGE": 1.3, "TREND_UP": 0.8, "TREND_DN": 0.5},
    "calendar_spread":   {"RANGE": 1.2, "LOWVOL": 1.1},
    "collar":            {"HIGHVOL": 1.4, "TREND_DN": 1.2},
}

_STRUCTURE_VOL_AFFINITY: Dict[str, Dict[str, float]] = {
    "long_call":        {"LOW_IV": 1.4, "NORMAL": 1.0, "HIGH_IV": 0.6},
    "long_put":         {"LOW_IV": 1.4, "NORMAL": 1.0, "HIGH_IV": 0.6},
    "iron_condor":       {"HIGH_IV": 1.5, "NORMAL": 1.0, "LOW_IV": 0.6},
    "iron_butterfly":    {"HIGH_IV": 1.4, "NORMAL": 1.0, "LOW_IV": 0.5},
    "short_straddle":    {"HIGH_IV": 1.5, "NORMAL": 0.8, "LOW_IV": 0.3},
    "long_straddle":     {"LOW_IV": 1.3, "NORMAL": 1.0, "HIGH_IV": 0.5},
    "long_strangle":     {"LOW_IV": 1.3, "NORMAL": 1.0, "HIGH_IV": 0.5},
    "bull_put_spread":   {"HIGH_IV": 1.3, "NORMAL": 1.0, "LOW_IV": 0.7},
    "bear_call_spread":  {"HIGH_IV": 1.3, "NORMAL": 1.0, "LOW_IV": 0.7},
}


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _group_stats(trades: List[Dict], key: str) -> Dict[str, Dict]:
    """Aggregate trade stats by a given key (structure, regime, vol_regime, etc.)."""
    stats: Dict[str, Dict] = {}
    for t in trades:
        g = t.get(key, "UNKNOWN")
        if g not in stats:
            stats[g] = {"count": 0, "wins": 0, "total_pnl": 0.0, "total_days": 0}
        stats[g]["count"] += 1
        stats[g]["wins"] += int(t.get("pnl_$", 0) > 0)
        stats[g]["total_pnl"] += float(t.get("cum_ret_%", 0))
        stats[g]["total_days"] += int(t.get("days_held", 0))

    for s in stats.values():
        cnt = s["count"]
        s["win_rate"] = s["wins"] / cnt if cnt else 0.0
        s["avg_pnl"] = s["total_pnl"] / cnt if cnt else 0.0
        s["avg_days"] = s["total_days"] / cnt if cnt else 0.0

    return stats


def _exit_reason_counts(trades: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Observations
# ──────────────────────────────────────────────────────────────────────────────

def _build_observations(
    trades: List[Dict],
    metrics: Dict,
    structure_stats: Dict[str, Dict],
    regime_stats: Dict[str, Dict],
    vol_stats: Dict[str, Dict],
    gex_stats: Dict[str, Dict],
    exit_stats: Dict[str, int],
) -> List[str]:
    obs: List[str] = []
    n = len(trades)
    if n == 0:
        obs.append("No trades were generated. The confidence gate may be too high "
                    "or the regime data insufficient.")
        return obs

    wins = sum(1 for t in trades if t.get("pnl_$", 0) > 0)
    wr = wins / n
    avg_hold = float(np.mean([t.get("days_held", 0) for t in trades]))

    obs.append(
        f"Executed **{n} trades** — {wins} winning, {n - wins} losing "
        f"(**{wr:.0%}** win rate), average hold **{avg_hold:.0f} days**."
    )

    # Best and worst structures
    qualified = {k: v for k, v in structure_stats.items() if v["count"] >= 2}
    if qualified:
        best = max(qualified, key=lambda k: qualified[k]["win_rate"])
        worst = min(qualified, key=lambda k: qualified[k]["win_rate"])
        if best != worst:
            obs.append(
                f"Best structure: **{best.replace('_', ' ')}** "
                f"({qualified[best]['win_rate']:.0%} win rate, "
                f"{qualified[best]['count']} trades). "
                f"Worst: **{worst.replace('_', ' ')}** "
                f"({qualified[worst]['win_rate']:.0%}, "
                f"{qualified[worst]['count']} trades)."
            )

    # Regime insights
    reg_q = {k: v for k, v in regime_stats.items() if v["count"] >= 2}
    if reg_q:
        best_r = max(reg_q, key=lambda k: reg_q[k]["win_rate"])
        worst_r = min(reg_q, key=lambda k: reg_q[k]["win_rate"])
        if best_r != worst_r:
            obs.append(
                f"Best regime for options: **{best_r}** "
                f"({reg_q[best_r]['win_rate']:.0%} win rate). "
                f"Weakest: **{worst_r}** ({reg_q[worst_r]['win_rate']:.0%})."
            )

    # IV regime insight
    vol_q = {k: v for k, v in vol_stats.items() if v["count"] >= 2}
    if vol_q:
        best_v = max(vol_q, key=lambda k: vol_q[k]["avg_pnl"])
        obs.append(
            f"Best IV regime: **{best_v}** "
            f"(avg return {vol_q[best_v]['avg_pnl']:+.1f}%, "
            f"{vol_q[best_v]['count']} trades)."
        )

    # Exit reasons
    if exit_stats:
        total = sum(exit_stats.values())
        reasons = ", ".join(f"{k}: {v} ({v/total:.0%})" for k, v in
                            sorted(exit_stats.items(), key=lambda x: -x[1]))
        obs.append(f"Exit breakdown: {reasons}.")

    # Sharpe
    sharpe = float(metrics.get("sharpe", 0))
    if sharpe >= 1.5:
        obs.append(f"Sharpe of **{sharpe:.2f}** indicates strong risk-adjusted performance.")
    elif sharpe >= 0.7:
        obs.append(f"Sharpe of **{sharpe:.2f}** is acceptable with room for improvement.")
    else:
        obs.append(f"Sharpe of **{sharpe:.2f}** signals inconsistent returns vs risk.")

    dd = abs(float(metrics.get("max_drawdown", 0)))
    if dd > 0.15:
        obs.append(f"Max drawdown **{dd:.0%}** warrants review of position sizing and exit rules.")

    return obs


# ──────────────────────────────────────────────────────────────────────────────
# Tweak generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_tweaks(
    trades: List[Dict],
    metrics: Dict,
    structure_stats: Dict[str, Dict],
    regime_stats: Dict[str, Dict],
    vol_stats: Dict[str, Dict],
    gex_stats: Dict[str, Dict],
    exit_stats: Dict[str, int],
    current_params: Optional[Dict] = None,
) -> List[OptionsTweak]:
    tweaks: List[OptionsTweak] = []
    n = len(trades)
    if n == 0:
        tweaks.append(OptionsTweak(
            id="lower_confidence_gate",
            category="confidence_gate",
            title="Lower Confidence Gate",
            rationale="No trades were generated. The confidence gate may be filtering out "
                      "all opportunities. Try lowering it to allow more regime states to trigger trades.",
            current_value=0.40,
            suggested_value=0.30,
            param_key="eng_min_conf",
            confidence="high",
        ))
        return tweaks

    wins = sum(1 for t in trades if t.get("pnl_$", 0) > 0)
    wr = wins / n
    dd = abs(float(metrics.get("max_drawdown", 0)))

    # ── Tweak 1: Confidence gate ────────────────────────────────────────
    if wr < 0.40 and n >= 4:
        tweaks.append(OptionsTweak(
            id="raise_confidence_gate",
            category="confidence_gate",
            title="Raise Confidence Gate",
            rationale=(
                f"Win rate of {wr:.0%} across {n} trades indicates the engine is entering "
                "marginal setups. A higher confidence gate focuses on high-conviction "
                "regime/IV/GEX alignments and filters noise."
            ),
            current_value=0.40,
            suggested_value=0.55,
            param_key="eng_min_conf",
            confidence="high",
        ))
    elif wr > 0.65 and n < 6:
        tweaks.append(OptionsTweak(
            id="lower_confidence_gate",
            category="confidence_gate",
            title="Lower Confidence Gate",
            rationale=(
                f"Strong {wr:.0%} win rate but only {n} trades — the gate may be too "
                "restrictive. Lowering it captures more valid setups."
            ),
            current_value=0.40,
            suggested_value=0.30,
            param_key="eng_min_conf",
            confidence="medium",
        ))

    # ── Tweak 2: Position sizing for high drawdown ──────────────────────
    if dd > 0.20:
        tweaks.append(OptionsTweak(
            id="reduce_position_size",
            category="position_size",
            title="Reduce Position Size",
            rationale=(
                f"Max drawdown of {dd:.0%} is elevated. Reducing position size from "
                "10% to 5-6% limits peak drawdown while preserving most upside."
            ),
            current_value=0.10,
            suggested_value=0.06,
            param_key="eng_pos_pct",
            confidence="high",
        ))

    # ── Tweak 3: IV premium adjustment ──────────────────────────────────
    vol_q = {k: v for k, v in vol_stats.items() if v["count"] >= 2}
    if vol_q:
        high_iv_stats = vol_q.get("HIGH_IV")
        low_iv_stats = vol_q.get("LOW_IV")
        if high_iv_stats and high_iv_stats["avg_pnl"] < -5.0:
            tweaks.append(OptionsTweak(
                id="lower_iv_premium",
                category="iv_premium",
                title="Lower IV Premium Multiplier",
                rationale=(
                    f"Trades in HIGH_IV conditions averaged {high_iv_stats['avg_pnl']:+.1f}% return. "
                    "The IV premium multiplier may be overestimating implied volatility, "
                    "causing overpaying for long premium. Try lowering to 1.05-1.10."
                ),
                current_value=1.15,
                suggested_value=1.08,
                param_key="eng_iv_prem",
                confidence="medium",
            ))

    # ── Tweak 4: Exit rules ─────────────────────────────────────────────
    stop_count = exit_stats.get("stop", 0)
    profit_count = exit_stats.get("profit", 0)
    expiry_count = exit_stats.get("expiry", 0)

    if stop_count > profit_count * 1.5 and n >= 4:
        tweaks.append(OptionsTweak(
            id="widen_stop_loss",
            category="exit_rules",
            title="Widen Stop Loss",
            rationale=(
                f"Stop-outs ({stop_count}) outnumber profit-takes ({profit_count}) by "
                f"{stop_count / max(profit_count, 1):.1f}x. The stop may be triggering on "
                "normal volatility swings. Try widening from 30% to 40%."
            ),
            current_value=0.30,
            suggested_value=0.40,
            param_key="eng_exit_stop",
            confidence="medium",
        ))

    if expiry_count > n * 0.4:
        tweaks.append(OptionsTweak(
            id="shorter_dte",
            category="dte_selection",
            title="Use Shorter DTE",
            rationale=(
                f"{expiry_count} of {n} trades ({expiry_count/n:.0%}) expired without hitting "
                "profit or stop. Shorter DTE forces faster resolution and reduces time decay drag."
            ),
            current_value=30,
            suggested_value=21,
            param_key="eng_dte",
            confidence="medium",
        ))

    # ── Tweak 5: Strategy mix ───────────────────────────────────────────
    struct_q = {k: v for k, v in structure_stats.items() if v["count"] >= 2}
    if struct_q:
        worst_struct = min(struct_q, key=lambda k: struct_q[k]["win_rate"])
        ws = struct_q[worst_struct]
        if ws["win_rate"] < 0.35:
            # Find a better structure for the dominant regime where this lost
            # Look at which regime this structure lost in most
            worst_regime_trades = [
                t for t in trades
                if t.get("structure") == worst_struct and t.get("pnl_$", 0) <= 0
            ]
            if worst_regime_trades:
                losing_regimes = {}
                for t in worst_regime_trades:
                    r = t.get("regime", "UNKNOWN")
                    losing_regimes[r] = losing_regimes.get(r, 0) + 1
                worst_losing_regime = max(losing_regimes, key=losing_regimes.get)

                # Find a structure with better affinity
                better = None
                best_affinity = 0.0
                for struct, affinities in _STRUCTURE_REGIME_AFFINITY.items():
                    if struct == worst_struct:
                        continue
                    a = affinities.get(worst_losing_regime, 1.0)
                    if a > best_affinity:
                        best_affinity = a
                        better = struct

                if better and best_affinity >= 1.2:
                    tweaks.append(OptionsTweak(
                        id=f"swap_{worst_struct}_for_{better}",
                        category="strategy_mix",
                        title=f"Consider {better.replace('_', ' ')} over {worst_struct.replace('_', ' ')}",
                        rationale=(
                            f"{worst_struct.replace('_', ' ')} had {ws['win_rate']:.0%} win rate, "
                            f"with losses concentrated in {worst_losing_regime}. "
                            f"{better.replace('_', ' ')} has stronger natural affinity for "
                            f"that regime (affinity score: {best_affinity:.1f}x)."
                        ),
                        current_value=worst_struct,
                        suggested_value=better,
                        param_key="strategy_preference",
                        confidence="medium",
                    ))

    # ── Tweak 6: GEX-aware caution ─────────────────────────────────────
    gex_q = {k: v for k, v in gex_stats.items() if v["count"] >= 2}
    flip_stats = gex_q.get("FLIP")
    if flip_stats and flip_stats["win_rate"] < 0.30:
        tweaks.append(OptionsTweak(
            id="avoid_gex_flip",
            category="confidence_gate",
            title="Increase Caution in GEX Flip Zones",
            rationale=(
                f"Trades entered during GEX flip conditions had only "
                f"{flip_stats['win_rate']:.0%} win rate. Consider raising the "
                "confidence gate when gex_flip_zone is active."
            ),
            current_value=0.40,
            suggested_value=0.60,
            param_key="eng_min_conf_flip",
            confidence="high",
        ))

    return tweaks


# ──────────────────────────────────────────────────────────────────────────────
# Summary and verdict
# ──────────────────────────────────────────────────────────────────────────────

def _build_summary_and_verdict(
    metrics: Dict,
    win_rate: float,
    n_trades: int,
) -> Tuple[str, str]:
    sharpe = float(metrics.get("sharpe", 0))
    total_return = float(metrics.get("total_return", 0))

    if n_trades == 0:
        return (
            "No trades were generated during this period. The confidence gate "
            "or regime conditions prevented the engine from entering any positions.",
            "neutral",
        )

    if sharpe >= 1.5 and total_return > 0.10 and win_rate >= 0.50:
        verdict = "strong"
        summary = (
            "The options engine delivered strong risk-adjusted returns with good "
            "structure-regime alignment. Tweaks below can refine edge further."
        )
    elif sharpe >= 0.7 and total_return > 0:
        verdict = "moderate"
        summary = (
            "The engine shows a real edge but has identifiable weaknesses in "
            "specific regimes or IV conditions. Targeted adjustments recommended."
        )
    elif total_return > 0:
        verdict = "weak"
        summary = (
            "Positive return achieved, but risk metrics need attention. "
            "Structure selection and exit timing are primary improvement areas."
        )
    else:
        verdict = "neutral"
        summary = (
            "The engine did not generate a positive return. Review confidence "
            "gate, IV premium assumptions, and strategy-regime alignment."
        )

    return summary, verdict


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def review_options_backtest(
    trade_log: List[Dict],
    metrics: Dict,
    ohlcv: Optional[pd.DataFrame] = None,
    current_params: Optional[Dict] = None,
) -> OptionsBacktestReview:
    """
    Generate a comprehensive options backtest review.

    Parameters
    ----------
    trade_log       : list of trade dicts from run_engine_backtest()
    metrics         : results dict from run_engine_backtest()
    ohlcv           : OHLCV DataFrame (for additional context)
    current_params  : current engine parameters for tweak suggestions

    Returns
    -------
    OptionsBacktestReview dataclass
    """
    n = len(trade_log)
    wins = sum(1 for t in trade_log if t.get("pnl_$", 0) > 0)
    wr = wins / n if n else 0.0
    avg_hold = float(np.mean([t.get("days_held", 0) for t in trade_log])) if trade_log else 0.0
    avg_pnl = float(np.mean([t.get("cum_ret_%", 0) for t in trade_log])) if trade_log else 0.0

    structure_stats = _group_stats(trade_log, "structure")
    regime_stats = _group_stats(trade_log, "regime")
    vol_stats = _group_stats(trade_log, "vol_regime")
    gex_stats = _group_stats(trade_log, "gex_regime")
    exit_stats = _exit_reason_counts(trade_log)

    observations = _build_observations(
        trade_log, metrics, structure_stats, regime_stats,
        vol_stats, gex_stats, exit_stats,
    )
    tweaks = _generate_tweaks(
        trade_log, metrics, structure_stats, regime_stats,
        vol_stats, gex_stats, exit_stats, current_params,
    )
    summary, verdict = _build_summary_and_verdict(metrics, wr, n)

    return OptionsBacktestReview(
        summary=summary,
        verdict=verdict,
        total_trades=n,
        win_rate=wr,
        avg_hold_days=avg_hold,
        avg_pnl_pct=avg_pnl,
        structure_stats=structure_stats,
        regime_stats=regime_stats,
        vol_regime_stats=vol_stats,
        gex_regime_stats=gex_stats,
        observations=observations,
        tweaks=tweaks,
        exit_reason_stats=exit_stats,
    )
