"""Legacy entry point for the Live Backtest Workbench.
Delegates to the modular app_streamlit.main module.
"""

from __future__ import annotations

from app_streamlit.main import main


if __name__ == "__main__":
    main()