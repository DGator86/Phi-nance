"""
Phibot Post-Backtest Review Engine
====================================
Analyses a completed backtest and generates a regime-aware mini-report
with actionable tweaks.

Philosophy — situational awareness, NOT overfitting:
  All suggestions are grounded in known indicator-regime affinities
  (e.g. "MACD excels in trending markets") rather than curve-fitting
  specific parameter values to the dataset under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Tweak:
    """A single actionable suggestion from Phibot."""
    id: str
    category: str       # blend_weight | signal_threshold | blend_method | add_indicator | position_sizing
    title: str
    rationale: str      # Regime-aware explanation (not overfitting justification)
    current_value: Any
    suggested_value: Any
    param_key: str      # Session-state key to update when the tweak is adopted
    confidence: str = "medium"  # high | medium | low


@dataclass
class BacktestReview:
    """Full Phibot review output."""
    summary: str
    verdict: str                        # strong | moderate | weak | neutral
    regime_stats: Dict[str, Dict]       # regime → {count, wins, win_rate, avg_pl}
    observations: List[str]
    tweaks: List[Tweak]
    total_trades: int
    win_rate: float
    avg_hold_bars: float


# ─── Regime Detection ─────────────────────────────────────────────────────────

_REGIME_DESCRIPTIONS: Dict[str, str] = {
    "TREND_UP": "uptrending markets",
    "TREND_DN": "downtrending markets",
    "RANGE": "range-bound markets",
    "HIGHVOL": "high-volatility markets",
    "LOWVOL": "low-volatility markets",
    "BREAKOUT_UP": "upside breakout conditions",
    "BREAKOUT_DN": "downside breakout conditions",
    "UNKNOWN": "mixed conditions",
}


def detect_market_regime(ohlcv: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Lightweight regime detector from OHLCV — no full regime engine required.

    Priority stack (most specific wins):
      BREAKOUT > TREND > HIGHVOL/LOWVOL > RANGE

    Returns a pd.Series of str regime labels aligned to ohlcv.index.
    """
    if ohlcv is None or len(ohlcv) < lookback * 2:
        idx = ohlcv.index if ohlcv is not None else pd.Index([])
        return pd.Series("RANGE", index=idx)

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # True Range → ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback).mean()
    atr_pct = (atr / close.replace(0, np.nan)).fillna(0)

    # Trend via rate-of-change
    roc = close.pct_change(lookback).fillna(0)

    # Volatility regime: ATR% vs its own rolling mean
    hist_window = min(60, max(lookback * 2, len(ohlcv) // 2))
    atr_ma = atr_pct.rolling(hist_window).mean().fillna(atr_pct.expanding().mean())
    vol_ratio = (atr_pct / atr_ma.replace(0, 1)).fillna(1.0)

    # Directional movement proxy (simplified ADX)
    dm_pos = (high - high.shift(1)).clip(lower=0).fillna(0)
    dm_neg = (low.shift(1) - low).clip(lower=0).fillna(0)
    atr_sum = tr.rolling(lookback).sum().replace(0, 1)
    di_pos = dm_pos.rolling(lookback).sum() / atr_sum
    di_neg = dm_neg.rolling(lookback).sum() / atr_sum
    di_sum = (di_pos + di_neg).replace(0, 1)
    adx_proxy = ((di_pos - di_neg).abs() / di_sum).rolling(lookback).mean().fillna(0)

    # Channel breakout reference
    upper_ch = high.rolling(lookback).max().shift(1)
    lower_ch = low.rolling(lookback).min().shift(1)

    # Build regime Series — layers applied bottom-up (later overwrites earlier)
    regime = pd.Series("RANGE", index=close.index)

    # Layer 1 — Volatility base
    regime[vol_ratio >= 1.5] = "HIGHVOL"
    regime[vol_ratio <= 0.55] = "LOWVOL"

    # Layer 2 — Trend override
    trend_strong = adx_proxy >= 0.22
    regime[trend_strong & (roc >= 0.02)] = "TREND_UP"
    regime[trend_strong & (roc <= -0.02)] = "TREND_DN"

    # Layer 3 — Breakout override (most specific signal)
    broke_up = upper_ch.notna() & (close >= upper_ch)
    broke_dn = lower_ch.notna() & (close <= lower_ch)
    regime[broke_up] = "BREAKOUT_UP"
    regime[broke_dn] = "BREAKOUT_DN"

    return regime.fillna("RANGE")


# ─── Trade Reconstruction ─────────────────────────────────────────────────────

def _reconstruct_trades(
    prediction_log: List[Dict],
    regime_series: Optional[pd.Series],
) -> List[Dict]:
    """
    Reconstruct BUY→SELL pairs from the bar-by-bar prediction log.
    Mirrors the position logic in phi/backtest/direct.py.
    """
    if not prediction_log:
        return []

    trades: List[Dict] = []
    in_trade = False
    entry_price = 0.0
    entry_date: Any = None
    entry_bar = 0

    for bar_idx, bar in enumerate(prediction_log):
        sig = bar.get("signal", "NEUTRAL")
        price = bar.get("price", 0.0)
        date = bar.get("date")

        if sig == "UP" and not in_trade and price > 0:
            in_trade = True
            entry_price = price
            entry_date = date
            entry_bar = bar_idx

        elif sig == "DOWN" and in_trade and price > 0:
            pnl_pct = (price - entry_price) / entry_price if entry_price else 0.0
            hold_bars = bar_idx - entry_bar

            # Regime at entry
            entry_regime = "UNKNOWN"
            if regime_series is not None and entry_date is not None:
                try:
                    ts = pd.Timestamp(entry_date)
                    if ts in regime_series.index:
                        entry_regime = str(regime_series[ts])
                    else:
                        idx = regime_series.index.get_indexer([ts], method="nearest")[0]
                        if idx >= 0:
                            entry_regime = str(regime_series.iloc[idx])
                except Exception:
                    pass

            trades.append(
                dict(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl_pct=pnl_pct,
                    win=pnl_pct > 0,
                    hold_bars=hold_bars,
                    regime=entry_regime,
                )
            )
            in_trade = False

    return trades


# ─── Regime Statistics ────────────────────────────────────────────────────────

def _compute_regime_stats(trades: List[Dict]) -> Dict[str, Dict]:
    """Aggregate trade stats by entry regime."""
    stats: Dict[str, Dict] = {}
    for t in trades:
        r = t.get("regime", "UNKNOWN")
        if r not in stats:
            stats[r] = {"count": 0, "wins": 0, "total_pl": 0.0}
        stats[r]["count"] += 1
        stats[r]["wins"] += int(t["win"])
        stats[r]["total_pl"] += t["pnl_pct"]

    for r, s in stats.items():
        cnt = s["count"]
        s["win_rate"] = s["wins"] / cnt if cnt else 0.0
        s["avg_pl"] = s["total_pl"] / cnt if cnt else 0.0

    return stats


# ─── Indicator-Regime Affinities (mirrors blender.py) ────────────────────────

_REGIME_INDICATOR_BOOST: Dict[str, Dict[str, float]] = {
    "RSI": {"TREND_UP": 1.2, "TREND_DN": 1.2, "RANGE": 1.3,
            "BREAKOUT_UP": 1.1, "BREAKOUT_DN": 1.1},
    "MACD": {"TREND_UP": 1.5, "TREND_DN": 1.5,
             "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3, "RANGE": 0.6},
    "Bollinger": {"RANGE": 1.4, "TREND_UP": 1.0, "TREND_DN": 1.0, "LOWVOL": 1.2},
    "Dual SMA": {"TREND_UP": 1.5, "TREND_DN": 1.5,
                 "BREAKOUT_UP": 1.3, "BREAKOUT_DN": 1.3, "RANGE": 0.5},
    "Mean Reversion": {"RANGE": 1.6, "LOWVOL": 1.2, "TREND_UP": 0.5, "TREND_DN": 0.5},
    "Breakout": {"BREAKOUT_UP": 1.5, "BREAKOUT_DN": 1.5,
                 "TREND_UP": 1.2, "TREND_DN": 1.2, "RANGE": 0.6},
    "Buy & Hold": {},
}


# ─── Tweak Generation ─────────────────────────────────────────────────────────

def _generate_tweaks(
    trades: List[Dict],
    regime_stats: Dict[str, Dict],
    indicators: Dict,
    blend_weights: Dict[str, float],
    blend_method: str,
    results: Dict,
) -> List[Tweak]:
    tweaks: List[Tweak] = []
    total_trades = len(trades)
    if total_trades == 0:
        return tweaks

    overall_win_rate = sum(t["win"] for t in trades) / total_trades
    dd = float(results.get("max_drawdown") or 0)

    # — Tweak 1: Signal threshold —
    if overall_win_rate < 0.42 and total_trades >= 4:
        tweaks.append(Tweak(
            id="threshold_up",
            category="signal_threshold",
            title="Raise Signal Threshold",
            rationale=(
                f"Win rate of {overall_win_rate:.0%} across {total_trades} trades indicates "
                "the strategy is acting on marginal signals. A higher entry threshold focuses "
                "on high-conviction setups and reduces noise trades in choppy conditions."
            ),
            current_value=0.15,
            suggested_value=0.20,
            param_key="phibot_signal_threshold",
            confidence="high",
        ))
    elif overall_win_rate > 0.62 and total_trades < 8:
        tweaks.append(Tweak(
            id="threshold_down",
            category="signal_threshold",
            title="Lower Signal Threshold",
            rationale=(
                f"Strong win rate ({overall_win_rate:.0%}) but only {total_trades} trades "
                "suggests the threshold may be too restrictive. Lowering it could capture "
                "more valid setups without significantly harming signal quality."
            ),
            current_value=0.15,
            suggested_value=0.12,
            param_key="phibot_signal_threshold",
            confidence="medium",
        ))

    # — Tweak 2: Boost indicator weight for worst-performing regime —
    qualified = {r: s for r, s in regime_stats.items() if s.get("count", 0) >= 2}
    if qualified:
        worst_regime = min(qualified, key=lambda r: qualified[r]["win_rate"])
        worst_stats = qualified[worst_regime]

        if worst_stats["win_rate"] < 0.42:
            regime_desc = _REGIME_DESCRIPTIONS.get(worst_regime, worst_regime)

            # Best indicator for this regime that is already in use
            best_ind = None
            best_boost = 0.0
            for ind_name in indicators:
                boost = _REGIME_INDICATOR_BOOST.get(ind_name, {}).get(worst_regime, 1.0)
                if boost > best_boost:
                    best_boost = boost
                    best_ind = ind_name

            if best_ind and best_boost >= 1.1:
                cur_w = blend_weights.get(best_ind, 0.5)
                sug_w = round(min(1.0, cur_w * 1.35), 2)
                if abs(sug_w - cur_w) > 0.04:
                    tweaks.append(Tweak(
                        id=f"boost_{best_ind.lower().replace(' ', '_')}",
                        category="blend_weight",
                        title=f"Boost {best_ind} Weight",
                        rationale=(
                            f"Most losses occurred in {regime_desc} "
                            f"(win rate: {worst_stats['win_rate']:.0%}). "
                            f"{best_ind} has a natural edge in {worst_regime} conditions — "
                            "increasing its weight improves signal quality during these market phases."
                        ),
                        current_value=round(cur_w, 2),
                        suggested_value=sug_w,
                        param_key=f"wt_{best_ind}",
                        confidence="medium",
                    ))

    # — Tweak 3: Switch to regime-weighted blending —
    if blend_method != "regime_weighted" and len(qualified) >= 2:
        win_rates = [s["win_rate"] for s in qualified.values()]
        dispersion = max(win_rates) - min(win_rates)
        if dispersion >= 0.28:
            tweaks.append(Tweak(
                id="use_regime_weighted",
                category="blend_method",
                title="Switch to Regime-Weighted Blending",
                rationale=(
                    f"Win rate varies by {dispersion:.0%} across different market regimes, "
                    "revealing significant situational differences. Regime-Weighted blending "
                    "automatically amplifies indicators proven in the current regime and dampens "
                    "those that underperform — adapting to market conditions in real time."
                ),
                current_value=blend_method,
                suggested_value="regime_weighted",
                param_key="blend_method",
                confidence="high",
            ))

    # — Tweak 4: Add a missing indicator for the worst regime —
    if qualified:
        worst_for_missing = min(qualified, key=lambda r: qualified[r]["win_rate"])
        if qualified[worst_for_missing]["win_rate"] < 0.40:
            regime_desc = _REGIME_DESCRIPTIONS.get(worst_for_missing, worst_for_missing)
            best_missing = None
            best_boost_m = 0.0

            for ind_name, boost_map in _REGIME_INDICATOR_BOOST.items():
                if ind_name in indicators or ind_name == "Buy & Hold":
                    continue
                boost = boost_map.get(worst_for_missing, 1.0)
                if boost > best_boost_m and boost >= 1.3:
                    best_boost_m = boost
                    best_missing = ind_name

            if best_missing:
                tweaks.append(Tweak(
                    id=f"add_{best_missing.lower().replace(' ', '_')}",
                    category="add_indicator",
                    title=f"Add {best_missing}",
                    rationale=(
                        f"Losses are concentrated in {regime_desc} — "
                        f"a regime where {best_missing} excels but is absent from your current stack. "
                        "Adding it provides broader coverage without overfitting, since the improvement "
                        "is driven by general regime-indicator dynamics rather than this specific dataset."
                    ),
                    current_value=None,
                    suggested_value=best_missing,
                    param_key="add_indicator",
                    confidence="medium",
                ))

    # — Tweak 5: Position sizing for high drawdown —
    if dd > 0.20:
        tweaks.append(Tweak(
            id="reduce_position_size",
            category="position_sizing",
            title="Reduce Position Sizing",
            rationale=(
                f"Max drawdown of {dd:.0%} is elevated. The current 95% capital deployment "
                "amplifies adverse regime periods. Reducing to ~70% limits peak drawdown "
                "while preserving the majority of upside in favorable market conditions."
            ),
            current_value=0.95,
            suggested_value=0.70,
            param_key="phibot_position_size",
            confidence="high",
        ))

    return tweaks


# ─── Observations ─────────────────────────────────────────────────────────────

def _build_observations(
    trades: List[Dict],
    regime_stats: Dict[str, Dict],
    results: Dict,
) -> List[str]:
    obs: List[str] = []
    total_trades = len(trades)

    if total_trades == 0:
        obs.append("No completed trades found in this backtest period.")
        return obs

    wins = sum(t["win"] for t in trades)
    losses = total_trades - wins
    win_rate = wins / total_trades
    avg_hold = float(np.mean([t["hold_bars"] for t in trades]))

    obs.append(
        f"Executed **{total_trades} trades** — {wins} winning, {losses} losing "
        f"(**{win_rate:.0%}** win rate)."
    )

    total_gains = sum(t["pnl_pct"] for t in trades if t["win"])
    total_losses_amt = abs(sum(t["pnl_pct"] for t in trades if not t["win"]))
    if total_losses_amt > 0:
        pf = total_gains / total_losses_amt
        obs.append(f"Profit factor: **{pf:.2f}x** (gross gains ÷ gross losses).")

    obs.append(f"Average holding period: **{avg_hold:.0f} bars**.")

    # Regime insights (only for regimes with ≥ 2 trades)
    qualified = {r: s for r, s in regime_stats.items() if s["count"] >= 2}
    if qualified:
        best_r = max(qualified, key=lambda r: qualified[r]["win_rate"])
        worst_r = min(qualified, key=lambda r: qualified[r]["win_rate"])
        b = qualified[best_r]
        w = qualified[worst_r]

        obs.append(
            f"Best regime: **{best_r}** ({_REGIME_DESCRIPTIONS.get(best_r, best_r)}) — "
            f"{b['win_rate']:.0%} win rate over {b['count']} trades."
        )
        if worst_r != best_r:
            obs.append(
                f"Weakest regime: **{worst_r}** ({_REGIME_DESCRIPTIONS.get(worst_r, worst_r)}) — "
                f"{w['win_rate']:.0%} win rate over {w['count']} trades."
            )

    sharpe = float(results.get("sharpe") or 0)
    if sharpe >= 1.5:
        obs.append(f"Sharpe ratio of **{sharpe:.2f}** reflects consistent risk-adjusted returns.")
    elif sharpe >= 0.7:
        obs.append(f"Sharpe ratio of **{sharpe:.2f}** is acceptable but leaves room for improvement.")
    else:
        obs.append(f"Sharpe ratio of **{sharpe:.2f}** points to inconsistent returns relative to risk.")

    dd = float(results.get("max_drawdown") or 0)
    if dd > 0.15:
        obs.append(
            f"Max drawdown of **{dd:.0%}** warrants reviewing position sizing or exit conditions."
        )

    return obs


# ─── Summary + Verdict ────────────────────────────────────────────────────────

def _build_summary_and_verdict(
    results: Dict,
    win_rate: float,
) -> Tuple[str, str]:
    sharpe = float(results.get("sharpe") or 0)
    total_return = float(results.get("total_return") or 0)

    if sharpe >= 1.5 and total_return > 0.10 and win_rate >= 0.50:
        verdict = "strong"
        summary = (
            "The strategy delivered strong risk-adjusted returns with consistent signal quality. "
            "Regime patterns are favourable — tweaks below can refine the edge further."
        )
    elif sharpe >= 0.7 and total_return > 0:
        verdict = "moderate"
        summary = (
            "The strategy shows a real edge but has identifiable regime weaknesses. "
            "Targeted adjustments to blending and indicator coverage could lift consistency."
        )
    elif total_return > 0:
        verdict = "weak"
        summary = (
            "Positive return achieved, but risk characteristics need attention. "
            "Signal timing and regime alignment are the primary improvement areas."
        )
    else:
        verdict = "neutral"
        summary = (
            "The strategy did not generate a positive return over this period. "
            "Review indicator selection, blending approach, and market regime coverage."
        )

    return summary, verdict


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def review_backtest(
    ohlcv: Optional[pd.DataFrame],
    results: Dict,
    prediction_log: List[Dict],
    indicators: Dict,
    blend_weights: Dict[str, float],
    blend_method: str,
    config: Dict,
) -> BacktestReview:
    """
    Generate a Phibot post-backtest review.

    All suggestions are grounded in regime-indicator dynamics, not
    parameter optimisation for the specific dataset under test.

    Parameters
    ----------
    ohlcv           : OHLCV DataFrame (same one used for the backtest)
    results         : dict returned by run_direct_backtest / options backtest
    prediction_log  : list of {date, symbol, signal, price} dicts
    indicators      : indicator config dict
    blend_weights   : current blend weights dict
    blend_method    : current blend method string
    config          : workbench config dict (symbols, dates, capital, …)

    Returns
    -------
    BacktestReview dataclass
    """
    # Regime detection
    regime_series: Optional[pd.Series] = None
    if ohlcv is not None and not ohlcv.empty and len(ohlcv) >= 40:
        try:
            regime_series = detect_market_regime(ohlcv)
        except Exception:
            regime_series = None

    # Trade reconstruction & regime stats
    trades = _reconstruct_trades(prediction_log, regime_series)
    regime_stats = _compute_regime_stats(trades)

    # Aggregate metrics
    total_trades = len(trades)
    win_rate = (sum(t["win"] for t in trades) / total_trades) if total_trades > 0 else 0.0
    avg_hold = float(np.mean([t["hold_bars"] for t in trades])) if trades else 0.0

    observations = _build_observations(trades, regime_stats, results)
    tweaks = _generate_tweaks(trades, regime_stats, indicators, blend_weights, blend_method, results)
    summary, verdict = _build_summary_and_verdict(results, win_rate)

    return BacktestReview(
        summary=summary,
        verdict=verdict,
        regime_stats=regime_stats,
        observations=observations,
        tweaks=tweaks,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_hold_bars=avg_hold,
    )
