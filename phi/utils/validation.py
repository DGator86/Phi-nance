"""Centralized input validation and sanitization helpers."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from phi.exceptions import ValidationError

_TICKER_RE = re.compile(r"^[A-Z0-9.^]+$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sanitize_ticker(raw_symbol: str) -> str:
    """Normalize and validate a ticker symbol."""
    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        raise ValidationError("Ticker symbol is required.")
    if "/" in symbol or "\\" in symbol:
        raise ValidationError("Ticker symbol cannot contain path separators.")
    if not _TICKER_RE.fullmatch(symbol):
        raise ValidationError("Ticker symbol must match ^[A-Z0-9.^]+$.")
    return symbol


def validate_date_bounds(value: date, *, name: str, min_year: int = 1970, max_future_days: int = 7) -> date:
    """Validate date is in a practical range."""
    if value.year < min_year:
        raise ValidationError(f"{name} is too early; must be >= {min_year}-01-01.")
    latest_allowed = date.today().toordinal() + max_future_days
    if value.toordinal() > latest_allowed:
        raise ValidationError(f"{name} cannot be that far in the future.")
    return value


def validate_positive_number(value: float | int, *, name: str, allow_zero: bool = False) -> float:
    """Validate numeric input as positive (or non-negative)."""
    numeric = float(value)
    if allow_zero:
        if numeric < 0:
            raise ValidationError(f"{name} must be >= 0.")
    elif numeric <= 0:
        raise ValidationError(f"{name} must be > 0.")
    return numeric


def sanitize_run_id(run_id: str) -> str:
    """Validate safe run identifier for use as directory segment."""
    candidate = str(run_id).strip()
    if not candidate:
        raise ValidationError("Run ID is required.")
    if not _RUN_ID_RE.fullmatch(candidate):
        raise ValidationError("Run ID must be alphanumeric with optional '-' or '_'.")
    return candidate


def resolve_safe_path(base_dir: Path, relative_name: str) -> Path:
    """Resolve a child path and ensure it stays under base_dir."""
    candidate = sanitize_run_id(relative_name)
    resolved_base = base_dir.resolve()
    resolved_path = (resolved_base / candidate).resolve()
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError as exc:
        raise ValidationError("Resolved path escapes allowed directory.") from exc
    return resolved_path
