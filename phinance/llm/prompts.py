"""Prompt template loader for HVE Core markdown prompts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from phinance.llm.utils import to_pretty_json, truncate_text

PROMPTS_DIR = Path(__file__).resolve().parents[2] / ".github/prompts"


def _resolve_prompt_path(name: str) -> Path:
    normalized = name if name.endswith(".prompt.md") else f"{name}.prompt.md"
    path = PROMPTS_DIR / normalized
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        parts = text.split("---\n", 2)
        if len(parts) == 3:
            return parts[2].lstrip()
    return text


def load_prompt(name: str, max_chars: int = 8_000, **kwargs: Any) -> str:
    """Load and render a prompt template using both HVE and brace placeholders."""
    path = _resolve_prompt_path(name)
    rendered = _strip_frontmatter(path.read_text(encoding="utf-8"))

    for key, raw_value in kwargs.items():
        value = truncate_text(to_pretty_json(raw_value), max_chars=max_chars)
        rendered = rendered.replace(f"${{input:{key}}}", value)
        rendered = rendered.replace(f"{{{key}}}", value)

    # Remove any unreplaced input placeholders to keep final prompt clean.
    rendered = re.sub(r"\$\{input:[^}]+\}", "[missing input]", rendered)
    return rendered.strip()
