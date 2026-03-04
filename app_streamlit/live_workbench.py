#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Live Backtest Workbench
===================================
Premium quant SaaS-grade backtesting platform.
Dark mode • Purple/Orange theme • Card-based layout • Step-by-step flow

Entry point:
    streamlit run app_streamlit/live_workbench.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bootstrap path so phi/ and regime_engine/ are importable ─────────────────
_ROOT = str(Path(__file__).parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("IS_BACKTESTING", "True")

import numpy as np
import pandas as pd
import streamlit as st

from app_streamlit.styles import (
    WORKBENCH_CSS,
    PALETTE,
    step_header,
    result_ribbon_html,
)
from phi.data.cache import (
    is_cached,
    load_dataset,
    save_dataset,
    list_cached_datasets,
    delete_dataset,
    dataset_id,
    clear_all_cache,
)
from phi.data.fetchers import fetch, TIMEFRAMES, dataset_summary, VENDORS
from phi.indicators.registry import (
    INDICATOR_REGISTRY,
    list_indicators,
    compute_signal,
    compute_all_signals,
)
from phi.blending.blender import Blender, BLEND_MODES, BLEND_LABELS, default_regime_weights
from phi.backtest.run_config import (
    RunConfig,
    IndicatorConfig,
    ExitRules,
    PositionSizing,
    OptionsConfig,
)
from phi.backtest.run_history import save_run, list_runs, load_run, compare_runs, delete_run


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phi-nance Workbench",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(WORKBENCH_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ss(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _set(key: str, val):
    st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v, precision=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):+.{precision}%}"
    except Exception:
        return "—"


def _fmt_num(v, precision=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{precision}f}"
    except Exception:
        return "—"


def _fmt_usd(v):
    if v is None:
        return "—"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"


def _metric_label(key: str) -> str:
    return {
        "roi":            "Total Return",
        "total_return":   "Total Return",
        "cagr":           "CAGR",
        "sharpe":         "Sharpe",
        "sortino":        "Sortino",
        "max_drawdown":   "Max Drawdown",
        "direction_accuracy": "Dir. Accuracy",
        "profit_factor":  "Profit Factor",
        "win_rate":       "Win Rate",
    }.get(key, key.title())


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
def render_banner():
    st.markdown("""
<div class="wb-banner">
  <h1>⚗ Phi-nance Live Backtest Workbench</h1>
  <p>Fetch · Indicator Blend · PhiAI Tune · Simulate · Analyze · Compare · Export</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════════

def render_step1() -> Optional[Dict[str, Any]]:
    """Returns dataset dict {ohlcv, meta, symbol, timeframe, vendor, initial_capital} or None."""

    st.markdown(step_header(1, "Dataset Builder",
                             "Fetch & cache historical OHLCV data"), unsafe_allow_html=True)

    # ── Main row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        mode = st.radio(
            "Trading Mode",
            ["Equities", "Options"],
            horizontal=True,
            key="wb_mode",
            help="Equities = direct stock trades.  Options = synthetic options simulation.",
        )
    with c2:
        symbol = st.text_input("Symbol", value=_ss("wb_symbol", "SPY"),
                                key="wb_symbol_inp", placeholder="SPY").upper().strip()
    with c3:
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0, key="wb_tf")
    with c4:
        vendor = st.selectbox("Data Vendor", VENDORS, index=0, key="wb_vendor")
    with c5:
        initial_cap = st.number_input(
            "Initial Capital ($)",
            value=_ss("wb_initial_cap", 100_000),
            min_value=1_000,
            step=5_000,
            key="wb_capital_inp",
            format="%d",
        )
        if initial_cap <= 0:
            st.error("Capital must be > 0")
            initial_cap = 100_000

    # ── Date range ────────────────────────────────────────────────────────────
    d1, d2 = st.columns(2)
    today = date.today()
    with d1:
        start_dt = st.date_input("Start Date", value=date(2021, 1, 1),
                                  max_value=today - timedelta(days=2), key="wb_start")
    with d2:
        end_dt = st.date_input("End Date", value=today - timedelta(days=1),
                                max_value=today, min_value=start_dt + timedelta(days=10),
                                key="wb_end")

    cached = is_cached(vendor, symbol, timeframe, start_dt, end_dt)

    # ── Action buttons ────────────────────────────────────────────────────────
    b1, b2, b3 = st.columns([1, 1, 2])
    fetch_clicked  = b1.button("⬇ Fetch & Cache Data", type="primary", use_container_width=True)
    cached_clicked = b2.button("📁 Use Cached Data",   use_container_width=True, disabled=not cached)

    if cached:
        b3.success(f"✓ Cached: {symbol}/{timeframe} {start_dt}→{end_dt}")
    else:
        b3.info("No cache for this dataset — click Fetch to download.")

    # ── Fetch ────────────────────────────────────────────────────────────────
    if fetch_clicked:
        with st.status(f"Fetching {symbol} {timeframe} from {vendor}...", expanded=True) as s:
            try:
                st.write(f"Downloading {symbol} ({start_dt} → {end_dt}) at {timeframe}...")
                df = fetch(symbol, start_dt, end_dt, timeframe, vendor)
                if df is None or df.empty:
                    s.update(label="No data returned — try a different vendor or date range.",
                             state="error")
                    return _ss("wb_dataset")
                p = save_dataset(df, vendor, symbol, timeframe, start_dt, end_dt)
                _set("wb_dataset", {
                    "ohlcv": df, "symbol": symbol, "timeframe": timeframe,
                    "vendor": vendor, "initial_capital": float(initial_cap),
                    "start": str(start_dt), "end": str(end_dt),
                    "dataset_id": dataset_id(vendor, symbol, timeframe, start_dt, end_dt),
                })
                _set("wb_symbol", symbol)
                _set("wb_initial_cap", initial_cap)
                rows = len(df)
                s.update(label=f"Cached {rows:,} bars → {p.name}", state="complete")
            except Exception as e:
                s.update(label=f"Failed: {e}", state="error")
                st.exception(e)

    # ── Use cached ───────────────────────────────────────────────────────────
    if cached_clicked and cached:
        df = load_dataset(vendor, symbol, timeframe, start_dt, end_dt)
        if df is not None:
            _set("wb_dataset", {
                "ohlcv": df, "symbol": symbol, "timeframe": timeframe,
                "vendor": vendor, "initial_capital": float(initial_cap),
                "start": str(start_dt), "end": str(end_dt),
                "dataset_id": dataset_id(vendor, symbol, timeframe, start_dt, end_dt),
            })
            _set("wb_symbol", symbol)
            _set("wb_initial_cap", initial_cap)
            st.success(f"Loaded {len(df):,} bars from cache.")
        else:
            st.error("Cache read failed. Re-fetch the data.")

    # ── Dataset summary ───────────────────────────────────────────────────────
    ds = _ss("wb_dataset")
    if ds is not None:
        df = ds["ohlcv"]
        sym = ds["symbol"]
        sm = dataset_summary(df, sym)

        # Update capital in session if user changed it
        ds["initial_capital"] = float(initial_cap)

        cols = st.columns(6)
        cols[0].metric("Symbol",    sym)
        cols[1].metric("Bars",      f"{sm.get('rows', 0):,}")
        cols[2].metric("First Bar", sm.get("first_bar", "")[:10])
        cols[3].metric("Last Bar",  sm.get("last_bar", "")[:10])
        cols[4].metric("Last Close", f"${sm.get('last_close', 0):,.2f}")
        ret = sm.get("total_return", 0)
        cols[5].metric("Buy & Hold", f"{ret:+.1%}", delta=f"{ret:+.1%}")

        with st.expander("Price chart", expanded=False):
            chart_df = df[["close"]].rename(columns={"close": f"{sym} Close"})
            st.line_chart(chart_df, use_container_width=True)

    return _ss("wb_dataset")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Indicator Selection
# ═══════════════════════════════════════════════════════════════════════════════

def render_step2(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns list of selected indicator dicts."""

    st.markdown(step_header(2, "Indicator Selection",
                             "Choose indicators · tune parameters · enable PhiAI auto-tune"),
                unsafe_allow_html=True)

    all_inds  = list_indicators()
    _ss("wb_selected_inds", [])
    _ss("wb_ind_params",    {})
    _ss("wb_ind_autotune",  {})
    _ss("wb_ind_enabled",   {})

    # ── Search / filter ───────────────────────────────────────────────────────
    search = st.text_input("Search indicators", placeholder="e.g. rsi, macd, bollinger...",
                           key="wb_ind_search").lower().strip()

    filtered = [
        n for n in all_inds
        if not search or
           search in n or
           search in INDICATOR_REGISTRY[n]["display_name"].lower() or
           search in INDICATOR_REGISTRY[n]["description"].lower()
    ]

    # ── Two-column layout: Available | Selected ───────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("**Available Indicators**")
        for name in filtered:
            info = INDICATOR_REGISTRY[name]
            is_sel = name in st.session_state.get("wb_selected_inds", [])
            with st.container():
                ic1, ic2 = st.columns([4, 1])
                with ic1:
                    st.markdown(
                        f"**{info['display_name']}** "
                        f"<span class='ind-type-badge'>Type {info['type']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(info["description"])
                with ic2:
                    lbl = "✓ Added" if is_sel else "＋ Add"
                    if st.button(lbl, key=f"add_{name}", use_container_width=True,
                                 type="primary" if not is_sel else "secondary",
                                 disabled=is_sel):
                        sel = list(st.session_state.get("wb_selected_inds", []))
                        if name not in sel:
                            sel.append(name)
                            _set("wb_selected_inds", sel)
                            # Initialize default params
                            params = st.session_state.get("wb_ind_params", {})
                            params[name] = {k: v["default"] for k, v in info["params"].items()}
                            _set("wb_ind_params", params)
                            enabled = st.session_state.get("wb_ind_enabled", {})
                            enabled[name] = True
                            _set("wb_ind_enabled", enabled)
                            st.rerun()
                st.divider()

    with right_col:
        selected_inds: List[str] = st.session_state.get("wb_selected_inds", [])
        if not selected_inds:
            st.info("No indicators selected. Add some from the left panel.")
        else:
            st.markdown(f"**Selected ({len(selected_inds)})**")
            remove_queue = []

            for name in selected_inds:
                info   = INDICATOR_REGISTRY[name]
                params = st.session_state.get("wb_ind_params", {}).get(name, {})
                autotune = st.session_state.get("wb_ind_autotune", {}).get(name, False)
                enabled  = st.session_state.get("wb_ind_enabled",  {}).get(name, True)

                card_cls = "ind-card" + (" auto-tuned" if autotune else " active" if enabled else "")
                st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)

                h1, h2, h3 = st.columns([3, 1, 1])
                with h1:
                    st.markdown(
                        f"**{info['display_name']}** "
                        f"<span class='ind-type-badge'>Type {info['type']}</span>",
                        unsafe_allow_html=True,
                    )
                with h2:
                    new_en = st.toggle("Enabled", value=enabled, key=f"en_{name}")
                    en_map = st.session_state.get("wb_ind_enabled", {})
                    en_map[name] = new_en
                    _set("wb_ind_enabled", en_map)
                with h3:
                    if st.button("✕", key=f"rm_{name}", help="Remove indicator"):
                        remove_queue.append(name)

                # Auto-tune toggle
                at_map = st.session_state.get("wb_ind_autotune", {})
                new_at = st.toggle(
                    "Auto-tune (PhiAI)", value=autotune, key=f"at_{name}",
                    help="PhiAI will optimize this indicator's parameters.",
                )
                at_map[name] = new_at
                _set("wb_ind_autotune", at_map)

                # Manual param tuning (expandable)
                if info["params"] and not new_at:
                    with st.expander("Manual Parameters", expanded=False):
                        p_map = st.session_state.get("wb_ind_params", {})
                        cur_p = p_map.get(name, {})
                        new_p = {}
                        pcols = st.columns(min(len(info["params"]), 3))
                        for idx, (pk, pspec) in enumerate(info["params"].items()):
                            with pcols[idx % len(pcols)]:
                                cur_val = cur_p.get(pk, pspec["default"])
                                if pspec["type"] == "int":
                                    new_p[pk] = st.slider(
                                        pspec["label"],
                                        min_value=int(pspec["min"]),
                                        max_value=int(pspec["max"]),
                                        value=int(cur_val),
                                        step=int(pspec.get("step", 1)),
                                        key=f"p_{name}_{pk}",
                                    )
                                else:
                                    new_p[pk] = st.slider(
                                        pspec["label"],
                                        min_value=float(pspec["min"]),
                                        max_value=float(pspec["max"]),
                                        value=float(cur_val),
                                        step=float(pspec.get("step", 0.1)),
                                        key=f"p_{name}_{pk}",
                                    )
                        p_map[name] = new_p
                        _set("wb_ind_params", p_map)

                st.markdown("</div>", unsafe_allow_html=True)

            # Remove dequeued
            if remove_queue:
                sel = [n for n in selected_inds if n not in remove_queue]
                _set("wb_selected_inds", sel)
                st.rerun()

    # Build result list
    result = []
    for name in st.session_state.get("wb_selected_inds", []):
        info = INDICATOR_REGISTRY[name]
        enabled   = st.session_state.get("wb_ind_enabled",  {}).get(name, True)
        autotune  = st.session_state.get("wb_ind_autotune", {}).get(name, False)
        params    = st.session_state.get("wb_ind_params",   {}).get(name, {})
        if not params:
            params = {k: v["default"] for k, v in info["params"].items()}
        result.append({
            "name":         name,
            "display_name": info["display_name"],
            "params":       params,
            "auto_tune":    autotune,
            "enabled":      enabled,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Blending Panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_step3(indicators: List[Dict]) -> Dict[str, Any]:
    """Returns blend config dict."""

    active = [i for i in indicators if i.get("enabled", True)]

    if len(active) < 2:
        return {"mode": "weighted_sum", "weights": {}, "regime_weights": {}}

    st.markdown(step_header(3, "Blending Panel",
                             "Choose how to combine multiple indicator signals"),
                unsafe_allow_html=True)

    mode = st.selectbox(
        "Blend Mode",
        BLEND_MODES,
        format_func=lambda m: BLEND_LABELS[m],
        key="wb_blend_mode",
    )

    weights:        Dict[str, float] = {}
    regime_weights: Dict[str, Dict[str, float]] = {}

    if mode == "weighted_sum":
        st.markdown("**Indicator Weights**")
        st.caption("Weights are automatically normalized to sum to 1.")
        wcols = st.columns(min(len(active), 4))
        for idx, ind in enumerate(active):
            with wcols[idx % len(wcols)]:
                w = st.slider(
                    ind["display_name"],
                    0.0, 2.0,
                    value=float(st.session_state.get(f"wb_w_{ind['name']}", 1.0)),
                    step=0.05,
                    key=f"wb_w_{ind['name']}",
                )
                weights[ind["name"]] = w

    elif mode == "voting":
        st.caption("Each enabled indicator gets one vote (±). Threshold for action = majority.")
        st.info("In voting mode, weights are equal. Enable/disable indicators in Step 2.")
        weights = {ind["name"]: 1.0 for ind in active}

    elif mode == "regime_weighted":
        st.caption(
            "Regime-aware: different weights per market regime. "
            "Defaults are auto-populated based on indicator type."
        )
        regime_weights = default_regime_weights([i["name"] for i in active])

        with st.expander("Regime Weight Matrix (advanced)", expanded=False):
            regimes = ["TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP",
                       "BREAKOUT_DN", "EXHAUST_REV", "LOWVOL", "HIGHVOL"]
            for regime in regimes:
                st.markdown(f"**{regime}**")
                rcols = st.columns(min(len(active), 4))
                for idx, ind in enumerate(active):
                    with rcols[idx % len(rcols)]:
                        default_w = regime_weights.get(regime, {}).get(ind["name"], 1.0)
                        rw = st.slider(
                            ind["display_name"][:12],
                            0.0, 2.0,
                            value=float(default_w),
                            step=0.1,
                            key=f"wb_rw_{regime}_{ind['name']}",
                        )
                        if regime not in regime_weights:
                            regime_weights[regime] = {}
                        regime_weights[regime][ind["name"]] = rw

    elif mode == "phiai":
        st.info(
            "PhiAI will automatically weight each indicator based on its individual "
            "backtest performance. Configure PhiAI in Step 4."
        )

    # ── Preview chart ─────────────────────────────────────────────────────────
    ds = _ss("wb_dataset")
    if ds is not None and active:
        with st.expander("Signal Preview (last 100 bars)", expanded=False):
            try:
                ohlcv = ds["ohlcv"]
                p_map = st.session_state.get("wb_ind_params", {})
                sigs  = compute_all_signals(
                    [i["name"] for i in active],
                    ohlcv.tail(200),
                    p_map,
                )
                blender = Blender(mode=mode, weights=weights if weights else None,
                                  regime_weights=regime_weights)
                blended = blender.blend(sigs)
                preview_df = sigs.copy()
                preview_df["blended"] = blended
                st.line_chart(preview_df.tail(100), use_container_width=True)
                st.caption("Individual signals (faint) + blended (bold)")
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")

    return {"mode": mode, "weights": weights, "regime_weights": regime_weights}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PhiAI Panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_step4(indicators: List[Dict]) -> Dict[str, Any]:
    """Returns PhiAI config dict."""

    st.markdown(step_header(4, "PhiAI Panel",
                             "Auto-tune parameters · auto-select indicators · regime-aware"),
                unsafe_allow_html=True)

    full_auto = st.toggle(
        "⚡ PhiAI Full Auto",
        value=_ss("wb_phiai_auto", False),
        key="wb_phiai_auto_toggle",
        help="PhiAI will auto-select the best indicators and tune all parameters.",
    )
    _set("wb_phiai_auto", full_auto)

    if full_auto:
        st.markdown('<div class="wb-card-purple">', unsafe_allow_html=True)
        st.markdown("**PhiAI will:**")
        st.markdown(
            "- Score each indicator individually using your chosen metric\n"
            "- Select the best subset (up to max indicators)\n"
            "- Run random-search to optimize parameters\n"
            "- Recommend blend weights based on scores\n"
            "- Apply drawdown constraint to prevent over-risky configs"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns(3)
        max_inds  = cc1.number_input("Max Indicators", 1, 8, value=3, key="wb_phiai_max_inds")
        risk_cap  = cc2.slider("Max Drawdown Cap", -0.80, -0.05, value=-0.30, step=0.05,
                               key="wb_phiai_dd_cap",
                               help="Reject configs with drawdown worse than this.")
        n_trials  = cc3.slider("Trials per Indicator", 10, 100, value=40, step=10,
                               key="wb_phiai_trials",
                               help="More trials = better optimization but slower.")
        no_short  = st.checkbox("No shorting", value=True, key="wb_phiai_no_short")

    else:
        # Show explanation of what PhiAI changed if it ran previously
        phiai_result = _ss("wb_phiai_result")
        if phiai_result:
            with st.expander("PhiAI Last Run Report", expanded=True):
                st.code(phiai_result.get("explanation", "No explanation available."),
                        language="text")

    return {
        "enabled":       full_auto,
        "max_indicators": int(_ss("wb_phiai_max_inds") if full_auto else 5),
        "max_dd_cap":    float(st.session_state.get("wb_phiai_dd_cap", -0.30)),
        "n_trials":      int(st.session_state.get("wb_phiai_trials", 40)),
        "no_short":      bool(st.session_state.get("wb_phiai_no_short", True)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Backtest Controls + RUN
# ═══════════════════════════════════════════════════════════════════════════════

def render_step5(
    dataset:    Dict[str, Any],
    indicators: List[Dict],
    blend_cfg:  Dict[str, Any],
    phiai_cfg:  Dict[str, Any],
):
    st.markdown(step_header(5, "Backtest Controls & Run",
                             "Configure trading rules · run · watch live progress"),
                unsafe_allow_html=True)

    trading_mode = st.session_state.get("wb_mode", "Equities").lower()

    left, right = st.columns(2, gap="large")

    # ── Left: position + exit ─────────────────────────────────────────────────
    with left:
        if trading_mode == "equities":
            st.markdown("**Equities Mode**")
            allow_short = st.checkbox("Allow Short Selling", value=False, key="wb_allow_short")
            pos_method  = st.selectbox("Position Sizing", ["Fixed % of Cash", "Fixed Shares"],
                                        key="wb_pos_method")
            pos_pct     = st.slider("Position Size (% of cash)", 0.10, 1.0, 0.95, 0.05,
                                     key="wb_pos_pct")

            st.markdown("**Exit Rules**")
            e1, e2 = st.columns(2)
            sl  = e1.slider("Stop Loss %",      0.0, 0.20, 0.0, 0.005, key="wb_sl",
                            help="0 = disabled", format="%.1%%")
            tp  = e2.slider("Take Profit %",    0.0, 0.30, 0.0, 0.005, key="wb_tp",
                            help="0 = disabled", format="%.1%%")
            ts  = e1.slider("Trailing Stop %",  0.0, 0.15, 0.0, 0.005, key="wb_ts",
                            help="0 = disabled", format="%.1%%")
            te  = e2.number_input("Time Exit (bars)", 0, 500, 0, key="wb_te",
                                   help="0 = disabled")
            sig_exit = st.checkbox("Exit on signal reversal", value=True, key="wb_sig_exit")

        else:
            st.markdown("**Options Mode**")
            structure  = st.selectbox(
                "Options Structure",
                ["long_call", "long_put", "debit_spread"],
                format_func=lambda s: {
                    "long_call":    "Long Call",
                    "long_put":     "Long Put",
                    "debit_spread": "Debit Spread",
                }[s],
                key="wb_opt_structure",
            )
            target_dte = st.slider("Target DTE (days to expiry)", 10, 90, 45, 5, key="wb_opt_dte")
            profit_pct = st.slider("Profit Exit %", 0.20, 1.00, 0.50, 0.05, key="wb_opt_profit",
                                    help="Exit when option gains this fraction of premium")
            stop_pct   = st.slider("Stop Loss % (of premium)", 0.50, 1.00, 1.00, 0.05,
                                    key="wb_opt_stop")
            pos_pct    = st.slider("Risk per Trade (% of capital)", 0.01, 0.15, 0.05, 0.01,
                                    key="wb_opt_pos")

    # ── Right: signal + evaluation ────────────────────────────────────────────
    with right:
        st.markdown("**Signal Parameters**")
        sig_threshold = st.slider(
            "Signal Threshold",
            0.01, 0.50, 0.10, 0.01,
            key="wb_sig_thresh",
            help="Minimum |signal| to trigger a trade.",
        )

        st.markdown("**Evaluation Metric**")
        eval_metric = st.selectbox(
            "Primary Metric",
            ["sharpe", "total_return", "cagr", "sortino", "max_drawdown",
             "direction_accuracy", "profit_factor", "win_rate"],
            format_func=_metric_label,
            key="wb_eval_metric",
        )

        st.markdown("**Run Description** (optional)")
        run_desc = st.text_input("Notes / tag this run", value="", key="wb_run_desc",
                                  placeholder="e.g. SPY daily RSI+MACD blend")

    st.divider()

    # ── Run / Stop buttons ────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
    run_clicked  = btn_col1.button("▶ Run Backtest", type="primary", use_container_width=True)
    stop_clicked = btn_col2.button("■ Stop", use_container_width=True)

    if stop_clicked:
        _set("wb_running", False)
        st.warning("Backtest stopped.")
        return

    if not run_clicked:
        return

    # ── Validate ─────────────────────────────────────────────────────────────
    active_inds = [i for i in indicators if i.get("enabled", True)]
    if not active_inds:
        st.error("Enable at least one indicator in Step 2.")
        return

    ohlcv = dataset.get("ohlcv")
    if ohlcv is None or ohlcv.empty:
        st.error("No dataset loaded. Complete Step 1 first.")
        return

    _set("wb_running", True)

    # ── Build RunConfig ───────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    cfg = RunConfig(
        run_id         = run_id,
        dataset_id     = dataset.get("dataset_id", ""),
        symbol         = dataset.get("symbol", ""),
        timeframe      = dataset.get("timeframe", "1D"),
        start_date     = dataset.get("start", ""),
        end_date       = dataset.get("end", ""),
        vendor         = dataset.get("vendor", "yfinance"),
        initial_capital = dataset.get("initial_capital", 100_000.0),
        trading_mode   = trading_mode,
        indicators     = [
            IndicatorConfig(
                name=i["name"], display_name=i["display_name"],
                params=i["params"], auto_tuned=i.get("auto_tune", False),
                enabled=True,
            )
            for i in active_inds
        ],
        blend_mode    = blend_cfg.get("mode", "weighted_sum"),
        blend_weights = blend_cfg.get("weights", {}),
        regime_weights = blend_cfg.get("regime_weights", {}),
        phiai_enabled = phiai_cfg.get("enabled", False),
        phiai_max_inds = phiai_cfg.get("max_indicators", 3),
        phiai_risk_cap = abs(float(phiai_cfg.get("max_dd_cap", -0.30))),
        exit_rules     = ExitRules(
            stop_loss_pct     = float(st.session_state.get("wb_sl", 0)) or None,
            take_profit_pct   = float(st.session_state.get("wb_tp", 0)) or None,
            trailing_stop_pct = float(st.session_state.get("wb_ts", 0)) or None,
            time_exit_bars    = int(st.session_state.get("wb_te", 0)) or None,
            signal_exit       = bool(st.session_state.get("wb_sig_exit", True)),
        ),
        position_sizing = PositionSizing(
            method      = "fixed_pct",
            pct_of_cash = float(st.session_state.get("wb_pos_pct", 0.95)),
            allow_short = bool(st.session_state.get("wb_allow_short", False)),
        ),
        options_config = OptionsConfig(
            structure       = str(st.session_state.get("wb_opt_structure", "long_call")),
            target_dte      = int(st.session_state.get("wb_opt_dte", 45)),
            profit_exit_pct = float(st.session_state.get("wb_opt_profit", 0.50)),
            stop_exit_pct   = float(st.session_state.get("wb_opt_stop", 1.00)),
        ),
        evaluation_metric = str(eval_metric),
        signal_threshold  = float(sig_threshold),
        description       = str(run_desc),
    )

    # ── Live backtest runner ──────────────────────────────────────────────────
    _run_live_backtest(cfg, ohlcv, active_inds, blend_cfg, phiai_cfg)
    _set("wb_running", False)


def _run_live_backtest(
    cfg:        RunConfig,
    ohlcv:      pd.DataFrame,
    active_inds: List[Dict],
    blend_cfg:  Dict[str, Any],
    phiai_cfg:  Dict[str, Any],
):
    """Execute backtest with live progress updates."""

    from phi.backtest.engine import BacktestEngine
    from phi.options.simulator import OptionsSimulator

    progress_bar = st.progress(0.0)
    status_area  = st.empty()
    log_area     = st.empty()
    logs: List[str] = []

    def _status(msg: str, pct: float):
        progress_bar.progress(min(pct, 1.0))
        status_area.markdown(
            f'<div class="wb-card"><b>{msg}</b></div>',
            unsafe_allow_html=True,
        )

    def _log(msg: str, kind: str = ""):
        ts = time.strftime("%H:%M:%S")
        logs.append(f'<p class="log-line {kind}">[{ts}] {msg}</p>')
        log_html = '<div class="log-console">' + "\n".join(logs[-30:]) + "</div>"
        log_area.markdown(log_html, unsafe_allow_html=True)

    try:
        # ── 1. Load data ──────────────────────────────────────────────────────
        _status("Step 1/6 — Loading data...", 0.0)
        _log(f"Dataset: {cfg.symbol} {cfg.timeframe}  |  {len(ohlcv):,} bars")
        _log(f"Initial capital: ${cfg.initial_capital:,.0f}")

        # ── 2. PhiAI (if enabled) ─────────────────────────────────────────────
        params_map = {i["name"]: i["params"] for i in active_inds}
        ind_names  = [i["name"] for i in active_inds]

        if phiai_cfg.get("enabled", False):
            _status("Step 2/6 — PhiAI auto-selecting & tuning...", 0.10)
            _log("PhiAI full auto mode enabled.", "info")
            try:
                from phi.phiai.tuner import phiai_full_auto
                from phi.indicators.registry import INDICATOR_REGISTRY as IND_REG

                tune_ranges_map = {n: IND_REG[n].get("tune_ranges", {}) for n in ind_names}
                bt_cfg_for_tuner = {
                    "initial_capital":  cfg.initial_capital,
                    "signal_threshold": cfg.signal_threshold,
                    "allow_short":      cfg.position_sizing.allow_short,
                    "position_pct":     cfg.position_sizing.pct_of_cash,
                    "timeframe":        cfg.timeframe,
                }

                def _phiai_prog(step, pct):
                    _status(f"Step 2/6 — PhiAI: {step}", 0.10 + 0.20 * pct)
                    _log(f"PhiAI: {step}", "info")

                result = phiai_full_auto(
                    ohlcv, ind_names, params_map, tune_ranges_map,
                    bt_cfg_for_tuner,
                    constraints={
                        "max_drawdown_cap": phiai_cfg.get("max_dd_cap", -0.30),
                        "no_short":         phiai_cfg.get("no_short", True),
                        "max_indicators":   phiai_cfg.get("max_indicators", 3),
                    },
                    metric     = cfg.evaluation_metric,
                    n_trials   = phiai_cfg.get("n_trials", 40),
                    progress_callback=_phiai_prog,
                )

                ind_names  = result["selected_indicators"]
                params_map = result["best_params"]
                blend_cfg  = {
                    **blend_cfg,
                    "mode":    "phiai",
                    "weights": result["blend_weights"],
                }
                _set("wb_phiai_result", result)
                _log(f"PhiAI selected: {', '.join(ind_names)}", "info")
            except Exception as e:
                _log(f"PhiAI failed ({e}), continuing with manual config.", "warn")

        # ── 3. Compute indicators ─────────────────────────────────────────────
        _status("Step 3/6 — Computing indicators...", 0.35)
        _log(f"Computing {len(ind_names)} indicator(s): {', '.join(ind_names)}")

        signals_df = compute_all_signals(ind_names, ohlcv, params_map)
        _log(f"Signals shape: {signals_df.shape}  |  NaN rows: {signals_df.isna().any(axis=1).sum()}")

        # ── 4. Blend signals ──────────────────────────────────────────────────
        _status("Step 4/6 — Blending signals...", 0.50)
        _log(f"Blend mode: {blend_cfg.get('mode', 'weighted_sum')}")

        blender = Blender(
            mode           = blend_cfg.get("mode", "weighted_sum"),
            weights        = blend_cfg.get("weights") or None,
            regime_weights = blend_cfg.get("regime_weights") or {},
        )

        # For phiai mode: inject metric scores if available
        if blend_cfg.get("mode") == "phiai":
            phiai_res = _ss("wb_phiai_result")
            if phiai_res:
                blender.metric_scores = phiai_res.get("metric_scores", {})

        blended_signal = blender.blend(signals_df)
        _log(f"Blended signal range: [{blended_signal.min():.3f}, {blended_signal.max():.3f}]")

        # ── 5. Run backtest ───────────────────────────────────────────────────
        _status("Step 5/6 — Simulating trades...", 0.60)

        backtest_cfg = {
            "initial_capital":   cfg.initial_capital,
            "signal_threshold":  cfg.signal_threshold,
            "allow_short":       cfg.position_sizing.allow_short,
            "position_pct":      cfg.position_sizing.pct_of_cash,
            "stop_loss_pct":     cfg.exit_rules.stop_loss_pct,
            "take_profit_pct":   cfg.exit_rules.take_profit_pct,
            "trailing_stop_pct": cfg.exit_rules.trailing_stop_pct,
            "time_exit_bars":    cfg.exit_rules.time_exit_bars,
            "signal_exit":       cfg.exit_rules.signal_exit,
            "timeframe":         cfg.timeframe,
        }

        prog_placeholder = [0.0]

        def _bt_progress(step: str, pct: float):
            _status(f"Step 5/6 — {step}", 0.60 + 0.20 * pct)

        if cfg.trading_mode == "options":
            sim = OptionsSimulator({
                **backtest_cfg,
                "structure":       cfg.options_config.structure,
                "target_dte":      cfg.options_config.target_dte,
                "profit_exit_pct": cfg.options_config.profit_exit_pct,
                "stop_exit_pct":   cfg.options_config.stop_exit_pct,
                "position_pct":    st.session_state.get("wb_opt_pos", 0.05),
            })
            opt_result = sim.run(ohlcv, blended_signal)
            # Wrap into a compatible BacktestResult-like object
            result_obj = _wrap_options_result(opt_result)
        else:
            engine = BacktestEngine(backtest_cfg)
            result_obj = engine.run(ohlcv, blended_signal, progress_callback=_bt_progress)

        for line in result_obj.run_log if hasattr(result_obj, "run_log") else []:
            _log(line)

        # ── 6. Compute metrics ────────────────────────────────────────────────
        _status("Step 6/6 — Computing metrics...", 0.85)
        metrics = result_obj.metrics

        end_cap   = result_obj.end_capital
        start_cap = cfg.initial_capital
        net_pnl   = end_cap - start_cap

        _log(f"Trades: {metrics.get('n_trades', 0):,}  |  "
             f"Win rate: {metrics.get('win_rate', 0):.1%}  |  "
             f"Sharpe: {metrics.get('sharpe', 0):.2f}")
        _log(f"End capital: ${end_cap:,.2f}  (net P/L: ${net_pnl:+,.2f})")

        # ── Save run ─────────────────────────────────────────────────────────
        _status("Saving run to history...", 0.95)
        run_data = {
            "metrics":     metrics,
            "end_capital": end_cap,
            "start_capital": start_cap,
            "equity_curve": result_obj.equity_curve,
            "signals":      blended_signal,
        }
        try:
            save_run(cfg, run_data, result_obj.trades)
            _log(f"Run saved: runs/{cfg.run_id}/", "info")
        except Exception as e:
            _log(f"Save failed: {e}", "warn")

        _set("wb_last_result", {
            "run_id":      cfg.run_id,
            "config":      cfg,
            "result":      result_obj,
            "signals_df":  signals_df,
            "blended":     blended_signal,
            "metrics":     metrics,
            "end_capital": end_cap,
        })

        _status("✓ Backtest complete!", 1.0)
        time.sleep(0.3)
        progress_bar.empty()
        status_area.empty()
        log_area.empty()
        st.rerun()

    except Exception as e:
        _status(f"Error: {e}", 1.0)
        _log(f"FATAL: {e}", "error")
        import traceback
        st.exception(e)


def _wrap_options_result(opt_result):
    """Wrap OptionsBacktestResult into a duck-typed object matching BacktestResult interface."""
    class _Wrapper:
        pass
    w = _Wrapper()
    w.equity_curve  = opt_result.equity_curve
    w.trades        = opt_result.trades
    w.metrics       = opt_result.metrics
    w.start_capital = opt_result.start_capital
    w.end_capital   = opt_result.end_capital
    w.signals       = pd.Series(dtype=float)
    w.positions     = pd.Series(dtype=float)
    w.run_log       = []
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_results():
    last = _ss("wb_last_result")
    if last is None:
        return

    result     = last["result"]
    metrics    = last["metrics"]
    cfg        = last["config"]
    signals_df = last.get("signals_df", pd.DataFrame())
    blended    = last.get("blended", pd.Series(dtype=float))
    run_id     = last["run_id"]
    end_cap    = last["end_capital"]
    start_cap  = cfg.initial_capital
    net_pnl    = end_cap - start_cap
    net_pnl_pct = net_pnl / (start_cap + 1e-10)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Results")

    # Eval metric value
    em   = cfg.evaluation_metric
    em_v = metrics.get(em, metrics.get("total_return", 0))
    em_lbl = _metric_label(em)
    if em in ("total_return", "cagr", "win_rate", "direction_accuracy", "max_drawdown"):
        em_str = f"{float(em_v):.2%}" if em_v is not None else "—"
    else:
        em_str = f"{float(em_v):.3f}" if em_v is not None else "—"

    st.markdown(
        result_ribbon_html(start_cap, end_cap, net_pnl, net_pnl_pct, em_lbl, em_str),
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Summary", "Equity Curve", "Trades", "Metrics", "Diagnostics"])

    # ── Summary ───────────────────────────────────────────────────────────────
    with tabs[0]:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total Return",  _fmt_pct(metrics.get("total_return")))
        rc2.metric("CAGR",          _fmt_pct(metrics.get("cagr")))
        rc3.metric("Sharpe",        _fmt_num(metrics.get("sharpe"), 2))
        rc4.metric("Max Drawdown",  _fmt_pct(metrics.get("max_drawdown")))

        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("Win Rate",      _fmt_pct(metrics.get("win_rate")))
        rc6.metric("Profit Factor", _fmt_num(metrics.get("profit_factor"), 2))
        rc7.metric("Trades",        f"{metrics.get('n_trades', 0):,}")
        rc8.metric("Dir. Accuracy", _fmt_pct(metrics.get("direction_accuracy")))

        st.markdown(f"**Run ID:** `{run_id}`  |  **Symbol:** {cfg.symbol}  |  "
                    f"**Timeframe:** {cfg.timeframe}  |  **Mode:** {cfg.trading_mode.title()}")
        st.caption(
            f"Indicators: {', '.join(i.display_name for i in cfg.indicators)}  |  "
            f"Blend: {cfg.blend_mode}  |  "
            f"PhiAI: {'on' if cfg.phiai_enabled else 'off'}"
        )

        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("Export Config (JSON)", key="exp_cfg", use_container_width=True):
                st.download_button(
                    "Download config.json",
                    data=cfg.to_json(),
                    file_name=f"run_{run_id}_config.json",
                    mime="application/json",
                )
        with ec2:
            if not result.trades.empty and st.button("Export Trades (CSV)",
                                                      key="exp_trades", use_container_width=True):
                st.download_button(
                    "Download trades.csv",
                    data=result.trades.to_csv(index=False),
                    file_name=f"run_{run_id}_trades.csv",
                    mime="text/csv",
                )

    # ── Equity Curve ──────────────────────────────────────────────────────────
    with tabs[1]:
        eq = result.equity_curve.to_frame("Portfolio Value ($)")
        if not blended.empty:
            # Normalize blended to overlay on equity chart
            pass
        st.line_chart(eq, use_container_width=True)

        # Drawdown chart
        if len(result.equity_curve) > 1:
            dd_series = (result.equity_curve - result.equity_curve.cummax()) / \
                        result.equity_curve.cummax()
            st.markdown("**Drawdown**")
            st.area_chart(dd_series.to_frame("Drawdown"), use_container_width=True)

    # ── Trades ────────────────────────────────────────────────────────────────
    with tabs[2]:
        if result.trades.empty:
            st.warning("No trades executed. Try lowering the signal threshold.")
        else:
            st.dataframe(result.trades, use_container_width=True, hide_index=True)

            if "pnl" in result.trades.columns:
                t1, t2 = st.columns(2)
                with t1:
                    pnl_hist = result.trades["pnl"]
                    pnl_df = pnl_hist.to_frame("Trade P&L ($)")
                    st.bar_chart(pnl_df, use_container_width=True)
                with t2:
                    cum_pnl = result.trades["pnl"].cumsum()
                    st.line_chart(cum_pnl.to_frame("Cumulative P&L ($)"),
                                  use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    with tabs[3]:
        metric_groups = [
            ("Returns", ["total_return", "cagr", "net_pnl", "net_pnl_pct"]),
            ("Risk",    ["max_drawdown", "volatility_annual", "sharpe", "sortino", "calmar"]),
            ("Trades",  ["n_trades", "win_rate", "profit_factor", "avg_win",
                         "avg_loss", "largest_win", "largest_loss", "avg_hold_bars"]),
            ("Accuracy", ["direction_accuracy"]),
            ("Capital", ["initial_capital", "end_capital"]),
        ]
        for group_name, keys in metric_groups:
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(keys))
            for col, k in zip(cols, keys):
                v = metrics.get(k)
                if v is None:
                    col.metric(k.replace("_", " ").title(), "—")
                elif k in ("total_return", "cagr", "win_rate", "max_drawdown",
                           "volatility_annual", "direction_accuracy", "net_pnl_pct"):
                    col.metric(k.replace("_", " ").title(), _fmt_pct(v))
                elif k in ("initial_capital", "end_capital", "net_pnl",
                           "avg_win", "avg_loss", "largest_win", "largest_loss"):
                    col.metric(k.replace("_", " ").title(), _fmt_usd(v))
                elif k in ("n_trades",):
                    col.metric(k.replace("_", " ").title(), f"{int(v):,}")
                else:
                    col.metric(k.replace("_", " ").title(), _fmt_num(v, 3))
            st.markdown("")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with tabs[4]:
        if not signals_df.empty:
            st.markdown("**Individual Indicator Signals**")
            st.line_chart(signals_df.tail(300), use_container_width=True)

        if not blended.empty:
            st.markdown("**Blended Signal**")
            st.line_chart(blended.tail(300).to_frame("Blended Signal"),
                          use_container_width=True)

        if hasattr(result, "positions") and not result.positions.empty:
            st.markdown("**Position Series (+1=long, -1=short, 0=flat)**")
            st.line_chart(result.positions.tail(300).to_frame("Position"),
                          use_container_width=True)

        # PhiAI explanation
        phiai_res = _ss("wb_phiai_result")
        if phiai_res and phiai_res.get("explanation"):
            with st.expander("PhiAI Explanation", expanded=True):
                st.code(phiai_res["explanation"], language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY TAB
# ═══════════════════════════════════════════════════════════════════════════════

def render_run_history():
    st.markdown("## Run History")

    runs = list_runs(limit=50)

    if not runs:
        st.info("No runs stored yet. Complete a backtest to see history.")
        return

    # Summary table
    display_cols = ["run_id", "created_at", "symbol", "timeframe", "trading_mode",
                    "indicators", "blend_mode", "phiai", "initial_capital",
                    "end_capital", "total_return", "sharpe", "max_drawdown",
                    "win_rate", "n_trades"]
    df_runs = pd.DataFrame(runs)
    avail_cols = [c for c in display_cols if c in df_runs.columns]
    st.dataframe(df_runs[avail_cols], use_container_width=True, hide_index=True)

    # Compare multiple runs
    run_ids = [r["run_id"] for r in runs]
    selected_ids = st.multiselect("Select runs to compare", run_ids, default=run_ids[:min(3, len(run_ids))],
                                   key="wb_compare_ids")

    if selected_ids and st.button("Compare Selected Runs", use_container_width=True):
        cmp_df = compare_runs(selected_ids)
        if not cmp_df.empty:
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Equity curve comparison
            st.markdown("**Equity Curves**")
            eq_data: Dict[str, pd.Series] = {}
            for rid in selected_ids:
                run = load_run(rid)
                if run and run.get("results"):
                    eq = run["results"].get("equity_curve")
                    if isinstance(eq, (pd.Series, pd.DataFrame)):
                        lbl = f"{run['results'].get('symbol', rid)} ({rid})"
                        eq_data[lbl] = eq if isinstance(eq, pd.Series) else eq.iloc[:, 0]
                    elif isinstance(eq, dict):
                        # Stored as dict
                        vals = list(eq.values())
                        if vals and isinstance(vals[0], list):
                            eq_data[rid] = pd.Series(vals[0])
            if eq_data:
                eq_df = pd.DataFrame(eq_data)
                st.line_chart(eq_df, use_container_width=True)

    # Delete
    with st.expander("Manage Runs", expanded=False):
        del_id = st.selectbox("Delete run", ["— select —"] + run_ids, key="wb_del_run")
        if del_id != "— select —":
            if st.button(f"Delete run {del_id}", key="wb_del_btn"):
                delete_run(del_id)
                st.success(f"Deleted run {del_id}")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def render_cache_manager():
    st.markdown("## Cache Manager")

    cached = list_cached_datasets()

    if not cached:
        st.info("No cached datasets. Use Step 1 to fetch and cache data.")
        return

    display = []
    for c in cached:
        display.append({
            "dataset_id": c.get("dataset_id", ""),
            "symbol":     c.get("symbol", ""),
            "timeframe":  c.get("timeframe", ""),
            "vendor":     c.get("vendor", ""),
            "start":      c.get("start", ""),
            "end":        c.get("end", ""),
            "rows":       c.get("rows", 0),
            "cached_at":  c.get("cached_at", "")[:19],
        })

    df_cache = pd.DataFrame(display)
    st.dataframe(df_cache, use_container_width=True, hide_index=True)

    if st.button("Clear All Cache", key="wb_clear_cache"):
        n = clear_all_cache()
        st.success(f"Cleared {n} cached datasets.")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    render_banner()

    # Top-level navigation tabs
    tab_wb, tab_hist, tab_cache = st.tabs([
        "⚗ Workbench",
        "📋 Run History",
        "💾 Cache Manager",
    ])

    with tab_wb:
        # ── Step 1 ────────────────────────────────────────────────────────────
        dataset = render_step1()

        if dataset is None:
            st.info("Complete Step 1 to continue.")
            # Show last result even if dataset not loaded
            render_results()
            return

        st.divider()

        # ── Step 2 ────────────────────────────────────────────────────────────
        indicators = render_step2(dataset)

        st.divider()

        # ── Step 3 (if 2+ indicators) ─────────────────────────────────────────
        active_inds = [i for i in indicators if i.get("enabled", True)]
        if len(active_inds) >= 2:
            blend_cfg = render_step3(indicators)
            st.divider()
        else:
            blend_cfg = {"mode": "weighted_sum", "weights": {}, "regime_weights": {}}

        # ── Step 4 ────────────────────────────────────────────────────────────
        phiai_cfg = render_step4(indicators)

        st.divider()

        # ── Step 5 ────────────────────────────────────────────────────────────
        render_step5(dataset, indicators, blend_cfg, phiai_cfg)

        # ── Results ───────────────────────────────────────────────────────────
        render_results()

    with tab_hist:
        render_run_history()

    with tab_cache:
        render_cache_manager()


if __name__ == "__main__":
    main()
