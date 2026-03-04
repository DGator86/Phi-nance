"""
app_streamlit/pages/__init__.py
================================

Page modules for the Phi-nance Streamlit app.
"""

from app_streamlit.pages.plugin_browser import render_plugin_browser
from app_streamlit.pages.live_trading_dashboard import render_live_trading_dashboard
from app_streamlit.pages.autonomous_pipeline_page import render_autonomous_pipeline

__all__ = [
    "render_plugin_browser",
    "render_live_trading_dashboard",
    "render_autonomous_pipeline",
]

# Phase 10 pages (optional import — only if modules are present)
try:
    from app_streamlit.pages.evolution_dashboard import render as render_evolution_dashboard
    from app_streamlit.pages.portfolio_backtest_page import render as render_portfolio_backtest
    __all__ += ["render_evolution_dashboard", "render_portfolio_backtest"]
except ImportError:
    pass

