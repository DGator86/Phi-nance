"""
frontend.legacy.dashboard — Original 6-tab dashboard (kept for reference).

This module preserves the original monolithic dashboard.py under the
legacy namespace. It is not used by the new multi-page Streamlit app.

To run the legacy dashboard:
    streamlit run frontend/legacy/dashboard.py
"""
# Re-export the original dashboard module unchanged.
# The actual code lives at /home/user/webapp/dashboard.py.
# This stub makes it accessible under the new package path.
import importlib.util
import sys
from pathlib import Path

_LEGACY_PATH = Path(__file__).resolve().parent.parent.parent / "dashboard.py"

if _LEGACY_PATH.exists():
    spec = importlib.util.spec_from_file_location("legacy_dashboard", _LEGACY_PATH)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["legacy_dashboard"] = _mod
        # Do NOT execute — just make importable for reference
