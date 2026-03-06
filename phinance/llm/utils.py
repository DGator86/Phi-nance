"""Utility helpers for LLM prompt rendering and context management."""

from __future__ import annotations

import json
from typing import Any


def to_pretty_json(value: Any) -> str:
    """Serialize a value for prompt injection."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def truncate_text(text: str, max_chars: int = 8_000) -> str:
    """Clip very long strings to avoid extreme token usage."""
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 60]}\n\n...[truncated {len(text) - max_chars} chars]"
