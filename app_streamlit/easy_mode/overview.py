"""Overview: overnight learning summary, recent runs, universe regimes."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from phi.options.regime_playbook import playbook_entry_for_label, quick_detailed_regime_from_ohlcv
from phi.run_config import RunHistory

from app_streamlit.easy_mode.constants import UNIVERSE
from app_streamlit.easy_mode.data import load_ohlcv
from app_streamlit.easy_mode.learning_summary import read_learning_summary


def render_overview() -> None:
    st.markdown('<p class="phinance-hero">Phi-nance overview</p>', unsafe_allow_html=True)
    st.caption("Last night’s learning, how we’re doing, and where each ticker sits in its regime.")

    summary = read_learning_summary()
    if summary:
        st.subheader("Overnight learning cycle")
        st.caption(f"Updated: {summary.get('updated_at', 'unknown')}")
        cols = st.columns(3)
        metrics = summary.get("metrics") or {}
        cols[0].metric("Cycle", str(summary.get("cycle", "nightly")))
        if "sharpe" in metrics:
            cols[1].metric("Sharpe (last run)", f"{float(metrics['sharpe']):.2f}")
        if "total_return" in metrics:
            cols[2].metric("Return (last run)", f"{float(metrics['total_return']) * 100:.1f}%")
        if summary.get("notes"):
            st.info(str(summary["notes"]))
        per = summary.get("per_ticker")
        if isinstance(per, dict) and per:
            st.dataframe(pd.DataFrame.from_dict(per, orient="index"), use_container_width=True)
    else:
        st.info(
            "No overnight report on disk yet. "
            "Point your cron job at **`data_cache/phi_nance_learning_summary.json`** "
            "(see `docs/easy_mode.md`) to show last night’s learning here."
        )

    st.divider()
    st.subheader("Recent backtests")
    runs = RunHistory().list_runs()[:8]
    if not runs:
        st.caption("No saved runs yet — run **Automatic backtest** once to populate history.")
    else:
        rows = []
        for r in runs:
            cfg = r.get("config") or {}
            res = r.get("results") or {}
            sym = (cfg.get("symbols") or ["—"])[0]
            rows.append({
                "When": r.get("run_id", "")[:15],
                "Symbol": sym,
                "Return %": round(float(res.get("total_return", 0) or 0) * 100, 2),
                "Sharpe": round(float(res.get("sharpe", 0) or 0), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Universe — current regime snapshot")
    st.caption("Quick read from price & volatility (not a saved ML model).")
    prog = st.progress(0.0)
    snap = []
    for i, sym in enumerate(UNIVERSE):
        prog.progress((i + 1) / max(len(UNIVERSE), 1))
        try:
            ohlcv = load_ohlcv(sym)
            lab = quick_detailed_regime_from_ohlcv(ohlcv)
            entry = playbook_entry_for_label(lab)
            close = float(ohlcv["close"].iloc[-1])
            prev = float(ohlcv["close"].iloc[-2]) if len(ohlcv) > 1 else close
            chg = (close - prev) / prev * 100 if prev else 0.0
            hint = (entry.summary[:80] + "…") if entry and len(entry.summary) > 80 else (entry.summary if entry else "")
            snap.append({
                "Ticker": sym,
                "Regime": lab,
                "Last close": round(close, 2),
                "1d %": round(chg, 2),
                "Playbook hint": hint,
            })
        except Exception as exc:  # noqa: BLE001
            snap.append({"Ticker": sym, "Regime": "—", "Last close": None, "1d %": None, "Playbook hint": str(exc)[:60]})
    prog.empty()
    st.dataframe(pd.DataFrame(snap), use_container_width=True, hide_index=True)
