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
    DATA_CACHE_DIR: Path = Path(os.getenv("DATA_CACHE_DIR", os.getenv("DATA_CACHE_ROOT", "./data_cache")))
    RUNS_DIR: Path = Path(os.getenv("RUNS_DIR", "./runs"))
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", "./logs"))
    DEBUG: bool = _env_bool("DEBUG", default=False)

    @property
    def DATA_CACHE_ROOT(self) -> Path:
        """Backward-compatible alias for ``DATA_CACHE_DIR``."""
        return self.DATA_CACHE_DIR

    def create_dirs(self) -> None:
        """Create configured runtime directories if they do not exist."""
        for directory in (self.DATA_CACHE_DIR, self.RUNS_DIR, self.LOGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
