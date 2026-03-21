"""Optional nightly learning-cycle JSON (written by your cron / external job)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phi.config import settings

from app_streamlit.easy_mode.constants import LEARNING_SUMMARY_FILENAME


def learning_summary_path() -> Path:
    return settings.DATA_CACHE_DIR / LEARNING_SUMMARY_FILENAME


def read_learning_summary() -> dict[str, Any] | None:
    p = learning_summary_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
