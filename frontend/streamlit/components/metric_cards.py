"""
frontend.streamlit.components.metric_cards
==========================================

Reusable HTML KPI metric card components.

Usage
-----
    from frontend.streamlit.components.metric_cards import render_kpi_row

    render_kpi_row([
        ("Total Return", "+12.4%", "positive"),
        ("Sharpe",        "1.85",  "positive"),
        ("Max DD",       "-8.3%",  "negative"),
    ])
"""

from __future__ import annotations

from typing import List, Tuple

import streamlit as st


def render_kpi_row(
    kpis: List[Tuple[str, str, str]],
    card_class: str = "phi-kpi-card",
) -> None:
    """Render a horizontal row of KPI metric cards.

    Parameters
    ----------
    kpis : list of (label, value, delta_type)
        delta_type: ``"positive"`` | ``"negative"`` | ``""``
    card_class : str — CSS class applied to each card
    """
    cards_html = ""
    for label, value, dtype in kpis:
        delta_cls = f"phi-kpi-delta-{dtype}" if dtype else ""
        cards_html += f"""
        <div class="{card_class}">
            <div class="phi-kpi-label">{label}</div>
            <div class="phi-kpi-value {delta_cls}">{value}</div>
        </div>"""
    st.markdown(
        f'<div class="phi-kpi-row">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def render_verdict_banner(verdict: str, summary: str) -> None:
    """Render a coloured Phibot verdict banner.

    Parameters
    ----------
    verdict : str — ``"strong"`` | ``"moderate"`` | ``"weak"`` | ``"neutral"``
    summary : str — summary text
    """
    colours = {
        "strong":   "#22c55e",
        "moderate": "#f59e0b",
        "weak":     "#f97316",
        "neutral":  "#94a3b8",
    }
    color = colours.get(verdict, "#94a3b8")
    st.markdown(
        f'<div style="padding:1rem;border-left:4px solid {color};'
        f'background:rgba(0,0,0,0.08);border-radius:4px;margin-bottom:1rem">'
        f'<b>Verdict: {verdict.upper()}</b> — {summary}'
        f"</div>",
        unsafe_allow_html=True,
    )
