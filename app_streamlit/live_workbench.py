"""Backward-compatible launcher for the modular live workbench app.

This module intentionally delegates to ``app_streamlit.main`` after the
refactor away from the previous monolithic implementation.
"""

from __future__ import annotations

from app_streamlit.main import main


if __name__ == "__main__":
    main()
