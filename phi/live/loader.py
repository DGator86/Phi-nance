"""Helpers for loading optimized live configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_latest_best_params(root: Path) -> Path | None:
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_live_config(config_path: Path | None = None, best_params_dir: Path | None = None) -> dict[str, Any]:
    if config_path is not None and config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    if best_params_dir is not None:
        latest = resolve_latest_best_params(best_params_dir)
        if latest is not None:
            return json.loads(latest.read_text(encoding="utf-8"))
    return {}
