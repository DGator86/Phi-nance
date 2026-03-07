"""Configuration settings for the ``phi`` package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings for logging/runtime behavior."""

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", "./logs"))
    DEBUG: bool = _env_bool("DEBUG", default=False)


settings = Settings()

