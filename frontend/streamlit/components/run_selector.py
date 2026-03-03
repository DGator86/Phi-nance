"""
frontend.streamlit.components.run_selector
==========================================

Reusable run-selector widget: displays a sortable history of saved
backtest runs and returns the selected run_id.

Usage
-----
    from frontend.streamlit.components.run_selector import render_run_selector
    run_id = render_run_selector()
    if run_id:
        run = RunHistory().load_run(run_id)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def render_run_selector(limit: int = 50) -> Optional[str]:
    """Display a run history table and return the selected run_id.

    Parameters
    ----------
    limit : int — maximum runs to display (default 50)

    Returns
    -------
    str or None — selected run_id, or None if nothing selected
    """
    try:
        from phinance.storage import RunHistory

        history = RunHistory()
        runs = history.list_runs(limit=limit)
    except Exception as exc:
        st.warning(f"Cannot load run history: {exc}")
        return None

    if not runs:
        st.info("No saved runs found.")
        return None

    rows = []
    for r in runs:
        rows.append({
            "Run ID":      r["run_id"],
            "Symbol":      ", ".join(r["config"].get("symbols", [])),
            "TF":          r["config"].get("timeframe", ""),
            "Mode":        r["config"].get("trading_mode", ""),
            "Return":      f"{r['results'].get('total_return', 0):.1%}",
            "Sharpe":      f"{r['results'].get('sharpe', 0):.2f}",
            "Max DD":      f"{r['results'].get('max_drawdown', 0):.1%}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    run_ids = [r["run_id"] for r in runs]
    selected = st.selectbox("Load a run", ["— select —"] + run_ids)
    if selected and selected != "— select —":
        return selected
    return None
