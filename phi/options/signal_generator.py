"""Build :class:`OptionsSignalCard` from OHLCV + playbook + MTF matrix + info indicators + optional PhiAI."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
import pandas as pd

from phi.indicators import (
    compute_entropy_signal,
    compute_fisher_information_signal,
    compute_mutual_info_signal,
)
from phi.logging import get_logger
from phi.options.engine import ENTRY_BIAS, STRATEGY_NAMES
from phi.options.regime_playbook import (
    playbook_entry_for_label,
    quick_detailed_regime_from_ohlcv,
    resolve_playbook_regime_key,
)
from phi.options.signal_card import OptionsSignalCard
from phi.phiai.auto_tune import load_best_params
from phi.regime.mtf_matrix import build_regime_matrix, confluence_score
from phi.regime.regime_definitions import infer_base_regime_from_prices

logger = get_logger(__name__)


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def _parse_trend_vol(composite: str) -> tuple[str, str]:
    u = composite.strip().upper()
    for prefix in ("RANGING", "BULL", "BEAR"):
        if u.startswith(prefix + "_"):
            return prefix, u[len(prefix) + 1 :]
    return "RANGING", "NORMAL_VOL"


def _desired_entry_bias(trend: str, vol: str) -> str:
    if trend == "RANGING" and "HIGH" in vol:
        return "volatile"
    if trend == "BULL":
        return "bullish"
    if trend == "BEAR":
        return "bearish"
    return "neutral"


def _display_name_to_engine_key(name: str) -> str:
    """Map playbook display names (e.g. 'Long Call') to engine keys (``long_call``)."""
    target = name.strip().lower()
    for k, v in STRATEGY_NAMES.items():
        if v.strip().lower() == target:
            return k
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_structure(allowed: list[str], bias: str) -> tuple[str, str]:
    """Return (structure_key, rationale)."""
    if not allowed:
        return "", "No playbook structures for this regime."
    keys: list[str] = [_display_name_to_engine_key(a) for a in allowed]

    for k in keys:
        if ENTRY_BIAS.get(k, "neutral") == bias:
            pretty = STRATEGY_NAMES.get(k, k)
            return k, f"Playbook allows {pretty}; matches {bias} bias."
    for k in keys:
        if ENTRY_BIAS.get(k, "neutral") == "neutral":
            pretty = STRATEGY_NAMES.get(k, k)
            return k, f"Fallback neutral structure: {pretty}."
    k0 = keys[0]
    return k0, f"First playbook structure: {STRATEGY_NAMES.get(k0, k0)} (bias mismatch — review)."


def _mtf_alignment(
    bias: str,
    confluence: float | None,
    *,
    bull_min: float,
    bear_max: float,
) -> tuple[str, str]:
    if confluence is None or (isinstance(confluence, float) and np.isnan(confluence)):
        return "N_A", "MTF matrix empty or unavailable for this bar spacing."
    if bias == "bullish":
        if confluence >= bull_min:
            return "WITH_TREND", f"MTF confluence {confluence:+.2f} supports bullish bias (≥{bull_min})."
        if confluence <= bear_max:
            return "AGAINST", f"MTF confluence {confluence:+.2f} disagrees with bullish idea."
        return "MIXED", f"MTF confluence {confluence:+.2f} is middling vs bullish bias."
    if bias == "bearish":
        if confluence <= -bull_min:
            return "WITH_TREND", f"MTF confluence {confluence:+.2f} supports bearish bias (≤{-bull_min})."
        if confluence >= -bear_max:
            return "AGAINST", f"MTF confluence {confluence:+.2f} disagrees with bearish idea."
        return "MIXED", f"MTF confluence {confluence:+.2f} is middling vs bearish bias."
    if bias == "volatile":
        return "MIXED", f"MTF confluence {confluence:+.2f} (long-vol plays: confirm with IV term structure separately)."
    return "MIXED", f"MTF confluence {confluence:+.2f} for neutral/range playbook."


def _info_last_bar(ohlcv: pd.DataFrame) -> tuple[dict[str, float], str]:
    out: dict[str, float] = {}
    notes: list[str] = []
    n = len(ohlcv)
    if n < 40:
        return out, "History too short for stable info metrics (need ~40+ bars)."
    try:
        e = compute_entropy_signal(ohlcv, window=min(20, n // 2))
        if len(e):
            out["entropy_signal"] = float(e.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("entropy_signal skipped: %s", exc)
    try:
        mi = compute_mutual_info_signal(ohlcv, window=min(20, n // 2))
        if len(mi):
            out["mutual_info_signal"] = float(mi.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("mutual_info skipped: %s", exc)
    try:
        fi = compute_fisher_information_signal(ohlcv, window=min(20, n // 2))
        if len(fi):
            out["fisher_information_signal"] = float(fi.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("fisher skipped: %s", exc)

    ent = out.get("entropy_signal")
    if ent is not None:
        if ent > 0.35:
            notes.append("Entropy elevated — path more random; tighten size or favor defined-risk.")
        elif ent < -0.25:
            notes.append("Entropy compressed — breakout risk; respect stops.")
    return out, " ".join(notes) if notes else ""


def _default_dataset_id(symbol: str, ohlcv: pd.DataFrame) -> str:
    base = f"{symbol.upper()}|{len(ohlcv)}|{ohlcv.index[0]}|{ohlcv.index[-1]}"
    return f"sig_{hashlib.sha1(base.encode()).hexdigest()[:12]}"


def build_options_signal_card(
    ohlcv: pd.DataFrame,
    *,
    symbol: str = "SPY",
    mtf_bull_min: float = 0.12,
    mtf_bear_threshold: float = 0.05,
    min_bars: int = 60,
    dataset_id: str | None = None,
) -> OptionsSignalCard:
    """Compose regime playbook, MTF matrix confluence, info indicators, and risk envelope."""
    if ohlcv is None or ohlcv.empty:
        return OptionsSignalCard(symbol=symbol, action="SKIP", reasoning=["No OHLCV data."])

    df = _normalize_ohlcv_columns(ohlcv)
    if len(df) < min_bars:
        return OptionsSignalCard(
            symbol=symbol,
            action="WAIT",
            reasoning=[f"Need at least {min_bars} bars; got {len(df)}."],
        )

    composite = quick_detailed_regime_from_ohlcv(df)
    entry = playbook_entry_for_label(composite)
    trend_price, _vol_token = _parse_trend_vol(composite)
    trend_ma = infer_base_regime_from_prices(df)
    bias = _desired_entry_bias(trend_price, _vol_token)

    reasoning: list[str] = [
        f"Composite regime (trend×vol): **{composite}**.",
        f"MA trend check: **{trend_ma}** (short vs long MA).",
    ]
    if trend_ma != trend_price:
        reasoning.append("Price trend tag disagrees with MA trend — treat as transition risk.")

    structure = ""
    struct_rat = ""
    if entry:
        structure, struct_rat = _pick_structure(list(entry.allowed_structures), bias)
        reasoning.append(
            f"Playbook: {entry.summary[:200]}{'…' if len(entry.summary) > 200 else ''}"
        )
    else:
        reasoning.append("No playbook row for this composite label — use manual discretion.")
        struct_rat = "Missing playbook entry."

    # MTF
    mtf_cols: list[str] = []
    mtf_conf: float | None = None
    mtf_align = "N_A"
    mtf_note = ""
    try:
        mat, meta = build_regime_matrix(df, min_resampled_bars=25)
        mtf_cols = list(mat.columns)
        if mat.shape[1] > 0:
            mtf_conf = float(confluence_score(mat).iloc[-1])
        trade_bias = ENTRY_BIAS.get(structure, bias) if structure else bias
        mtf_align, mtf_note = _mtf_alignment(
            trade_bias,
            mtf_conf,
            bull_min=mtf_bull_min,
            bear_max=mtf_bear_threshold,
        )
        skipped = meta.get("skipped") if isinstance(meta, dict) else {}
        if skipped:
            reasoning.append(f"MTF skipped some rules: {len(skipped)} (see desk expander).")
    except Exception as exc:  # noqa: BLE001
        mtf_note = f"MTF matrix failed: {exc}"
        reasoning.append(mtf_note)

    # Info theory
    info, info_notes = _info_last_bar(df)
    if info:
        reasoning.append(
            "Info snapshot (last bar): "
            + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(info.items()))
        )

    # PhiAI promotion
    dsid = dataset_id or _default_dataset_id(symbol, df)
    phiai_payload = load_best_params(dsid)
    phiai_ok = phiai_payload is not None
    if phiai_ok:
        reasoning.append(
            f"PhiAI promoted params on file for dataset `{dsid}` "
            f"({phiai_payload.get('metric')}={phiai_payload.get('best_value')})."
        )
    else:
        reasoning.append(
            f"No PhiAI promotion at `{dsid}` — run expert PhiAI sweep to populate `DATA_CACHE_DIR/phiai_best_params/`."
        )

    # Action gate
    action: Literal["ENTER", "WAIT", "SKIP"] = "ENTER"
    if not structure:
        action = "SKIP"
    elif mtf_align == "AGAINST" and bias in ("bullish", "bearish"):
        action = "WAIT"
        reasoning.append("Action lowered to WAIT: MTF confluence against directional bias.")
    elif trend_ma != trend_price and bias in ("bullish", "bearish"):
        action = "WAIT"
        reasoning.append("Action lowered to WAIT: MA vs regime trend mismatch.")

    dte_min = int(entry.dte_days_min) if entry else 14
    dte_max = int(entry.dte_days_max) if entry else 90
    d_lo, d_hi = (float(entry.delta_band[0]), float(entry.delta_band[1])) if entry else (0.35, 0.55)
    max_risk = float(entry.max_risk_pct_portfolio) if entry else 0.04

    trigger = (
        f"When underlying aligns with {STRATEGY_NAMES.get(structure, structure)}: "
        f"use {dte_min}-{dte_max} DTE, delta between {d_lo:.2f} and {d_hi:.2f}."
    )

    pb_key = resolve_playbook_regime_key(composite)
    return OptionsSignalCard(
        symbol=symbol.upper(),
        composite_regime=composite,
        playbook_regime_key=pb_key,
        action=action,
        structure=structure,
        structure_rationale=struct_rat,
        entry_trigger=trigger,
        target_exit_pct=0.50,
        stop_exit_pct=1.00,
        dte_days_min=dte_min,
        dte_days_max=dte_max,
        delta_band_low=d_lo,
        delta_band_high=d_hi,
        max_risk_pct_portfolio=max_risk,
        mtf_timeframes_present=mtf_cols,
        mtf_confluence=mtf_conf,
        mtf_alignment=mtf_align,
        mtf_notes=mtf_note,
        info_metrics=info,
        info_notes=info_notes,
        reasoning=reasoning,
        phiai_dataset_id=dsid,
        phiai_promoted=bool(phiai_ok),
        phiai_metric=str(phiai_payload.get("metric")) if phiai_payload else None,
        phiai_best_value=float(phiai_payload["best_value"]) if phiai_payload and "best_value" in phiai_payload else None,
    )
