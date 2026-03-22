"""Central hub UI: regimes, options playbook, and navigation hints for the workbench."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from phi.data import fetch_ohlcv_uw_then_yf
from phi.options.regime_playbook import (
    get_default_options_regime_playbook,
    playbook_entry_for_label,
    quick_detailed_regime_from_ohlcv,
)
from phi.options.signal_generator import build_options_signal_card


@st.cache_data(ttl=600, show_spinner=False)
def _load_ohlcv_for_signal(symbol: str, days: int, day_key: str) -> tuple[pd.DataFrame, str]:
    """OHLCV + which vendor supplied it (``unusual_whales`` preferred, else ``yfinance``)."""
    sym = symbol.strip().upper()
    end = date.today()
    start = end - timedelta(days=max(days, 80))
    return fetch_ohlcv_uw_then_yf(sym, start.isoformat(), end.isoformat(), timeframe="1D")


def render_trading_desk() -> None:
    st.title("Trading desk")
    st.markdown(
        "Use this page as the **front door**: understand where you are in the market, "
        "which options playbooks apply, then jump to the workbench to backtest."
    )

    st.subheader("Options signal card (entry · target · stop · MTF · regime · info · PhiAI)")
    st.caption(
        "OHLCV loads **Unusual Whales first**, then **yfinance** (same as easy mode). "
        "With `UNUSUAL_WHALES_API_KEY`, the card also pulls **ATM chain + flow** for the suggested structure. "
        "Not financial advice — validate in Backtest Workbench before risking capital."
    )
    c_sym, c_days, c_go = st.columns([2, 1, 1])
    with c_sym:
        sig_sym = st.text_input("Symbol", value="SPY", key="desk_signal_symbol")
    with c_days:
        sig_days = st.number_input("Lookback days", min_value=120, max_value=2000, value=420, step=30, key="desk_signal_days")
    with c_go:
        st.write("")
        st.write("")
        run_sig = st.button("Build signal", type="primary", key="desk_build_signal")

    if run_sig:
        try:
            ohl, v_used = _load_ohlcv_for_signal(sig_sym, int(sig_days), date.today().isoformat())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            ohl = None
            v_used = ""
        if ohl is not None:
            card = build_options_signal_card(ohlcv=ohl, symbol=sig_sym, ohlcv_vendor=v_used)
            st.session_state["desk_options_signal"] = card.model_dump()

    dumped = st.session_state.get("desk_options_signal")
    if dumped:
        act = dumped.get("action", "—")
        color = {"ENTER": "#22c55e", "WAIT": "#eab308", "SKIP": "#ef4444"}.get(act, "#94a3b8")
        st.markdown(
            f'<div style="background:{color}22;border:1px solid {color}66;border-radius:12px;padding:14px;margin:8px 0">'
            f'<div style="font-size:0.8rem;opacity:0.9">Action</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:{color}">{act}</div></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Structure", str(dumped.get("structure") or "—"))
        m2.metric("Target (premium %)", f"+{float(dumped.get('target_exit_pct', 0))*100:.0f}%")
        m3.metric("Stop (premium %)", f"-{float(dumped.get('stop_exit_pct', 0))*100:.0f}%")
        mc = dumped.get("mtf_confluence")
        m4.metric("MTF confluence", f"{mc:+.2f}" if mc is not None else "—")
        st.caption(f"OHLCV vendor: **{dumped.get('ohlcv_vendor') or 'unknown'}**")
        st.write("**Composite regime:**", f"`{dumped.get('composite_regime', '')}`")
        st.write("**MTF alignment:**", dumped.get("mtf_alignment"), "—", dumped.get("mtf_notes", ""))
        st.write("**Entry / sizing envelope:**", dumped.get("entry_trigger", ""))
        st.caption(
            f"DTE {dumped.get('dte_days_min')}–{dumped.get('dte_days_max')} · "
            f"delta {dumped.get('delta_band_low')}-{dumped.get('delta_band_high')} · "
            f"max book risk {float(dumped.get('max_risk_pct_portfolio', 0))*100:.1f}%"
        )
        if dumped.get("structure_rationale"):
            st.info(dumped["structure_rationale"])
        st.markdown("**Regime & context (reasoning)**")
        for line in dumped.get("reasoning") or []:
            st.markdown(f"- {line}")
        info = dumped.get("info_metrics") or {}
        if info:
            st.markdown("**Information theory (last bar)**")
            st.json(info)
            if dumped.get("info_notes"):
                st.caption(dumped["info_notes"])
        uw = dumped.get("unusual_whales_snapshot") or {}
        if uw:
            with st.expander("Unusual Whales (ATM chain + flow)", expanded=False):
                st.json(uw)
        with st.expander("MTF columns present", expanded=False):
            st.write(dumped.get("mtf_timeframes_present") or [])
        ph = dumped.get("phiai_promoted")
        st.caption(
            f"PhiAI dataset id: `{dumped.get('phiai_dataset_id')}` — "
            f"promoted: **{ph}**"
            + (
                f" ({dumped.get('phiai_metric')}={dumped.get('phiai_best_value')})"
                if ph
                else ""
            )
        )

    st.subheader("Quick orientation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Data & context**")
        st.caption("Backtest Workbench → vendor **unusual_whales** for options (Greeks + flow).")
    with c2:
        st.markdown("**2. Regime**")
        st.caption("Sidebar → enable regime-aware blending, train or load a detector, optional use session series.")
    with c3:
        st.markdown("**3. Options**")
        st.caption("Trading mode **options**; playbook expander in the form shows structures per regime row.")

    st.divider()

    results = st.session_state.get("results") or {}
    ohlcv = results.get("ohlcv")
    if ohlcv is not None and hasattr(ohlcv, "empty") and not ohlcv.empty:
        try:
            q = quick_detailed_regime_from_ohlcv(ohlcv)
            entry = playbook_entry_for_label(q)
            st.subheader("Snapshot from last run (price-based regime)")
            st.success(f"**Inferred composite regime:** `{q}`")
            if entry:
                st.write(entry.summary)
                st.caption("Structures: " + ", ".join(entry.allowed_structures[:6]) + " …")
        except Exception:  # noqa: BLE001
            st.info("Could not infer regime from cached results.")

    st.subheader("Full options regime playbook")
    pb = get_default_options_regime_playbook()
    st.caption(f"Playbook version **{pb.version}** — override with env `PHINANCE_OPTIONS_PLAYBOOK` (JSON path).")
    for key in sorted(pb.regimes.keys())[:12]:
        e = pb.regimes[key]
        with st.expander(f"{key} — {e.display_name}", expanded=False):
            st.write(e.summary)
            st.caption(", ".join(e.allowed_structures))
    if len(pb.regimes) > 12:
        st.caption(f"_… and {len(pb.regimes) - 12} more rows (see Backtest Workbench expander or docs)._")

    st.subheader("Regime transition map")
    for t in pb.transitions:
        st.markdown(
            f"**{t.from_regime} → {t.to_regime}**  \n"
            f"- {t.trigger}  \n"
            f"- *{t.action}*"
        )

    st.divider()
    st.info(
        "Switch sidebar → **Backtest Workbench** to configure symbols, run an options backtest, "
        "and view **Attributed PnL by regime** when regime blending is enabled."
    )
