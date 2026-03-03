"""
frontend.streamlit.utils — Streamlit-specific helpers.

Functions
---------
  inject_css(path)     — Inject a CSS file via st.markdown
  session_get(key, d)  — Safe session_state getter with default
  session_set(**kw)    — Bulk-update session state
  format_currency(v)   — Format a float as "$1,234.56"
  format_pct(v)        — Format a float as "+12.3%"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def inject_css(path: Path) -> None:
    """Inject a CSS file into the Streamlit page.

    Parameters
    ----------
    path : Path — absolute path to .css file
    """
    if path.exists():
        css = path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def session_get(key: str, default: Any = None) -> Any:
    """Get a value from ``st.session_state`` with a fallback default."""
    return st.session_state.get(key, default)


def session_set(**kwargs: Any) -> None:
    """Bulk-update ``st.session_state``."""
    for k, v in kwargs.items():
        st.session_state[k] = v


def format_currency(value: float, prefix: str = "$") -> str:
    """Format a number as currency string.

    Example
    -------
        format_currency(123456.78)  →  "$123,456.78"
    """
    return f"{prefix}{value:,.2f}"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a fractional value as percentage with sign.

    Example
    -------
        format_pct(0.123)   →  "+12.3%"
        format_pct(-0.045)  →  "-4.5%"
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def metric_delta_type(value: float, inverse: bool = False) -> str:
    """Return ``"positive"`` or ``"negative"`` for CSS styling.

    Parameters
    ----------
    value   : float — metric value
    inverse : bool  — when True, negative value is "positive" (e.g. drawdown)
    """
    positive = value >= 0
    if inverse:
        positive = not positive
    return "positive" if positive else "negative"
