"""Central hub UI: regimes, options playbook, and navigation hints for the workbench."""

from __future__ import annotations

import streamlit as st

from phi.options.regime_playbook import (
    get_default_options_regime_playbook,
    playbook_entry_for_label,
    quick_detailed_regime_from_ohlcv,
)


def render_trading_desk() -> None:
    st.title("Trading desk")
    st.markdown(
        "Use this page as the **front door**: understand where you are in the market, "
        "which options playbooks apply, then jump to the workbench to backtest."
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
