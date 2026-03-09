"""Configuration settings for the ``phi`` package."""

from __future__ import annotations

from phi.logging import get_logger

logger = get_logger(__name__)

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return tuple(default)
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings for logging/runtime behavior."""

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATA_CACHE_DIR: Path = Path(os.getenv("DATA_CACHE_DIR", os.getenv("DATA_CACHE_ROOT", "./data_cache")))
    RUNS_DIR: Path = Path(os.getenv("RUNS_DIR", "./runs"))
    REGIME_MODELS_DIR: Path = Path(
        os.getenv(
            "REGIME_MODELS_DIR",
            str(Path(os.getenv("RUNS_DIR", "./runs")) / "regime_models"),
        )
    )
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", "./logs"))
    DEBUG: bool = _env_bool("DEBUG", default=False)
    PHIAI_DEFAULT_N_TRIALS: int = int(os.getenv("PHIAI_DEFAULT_N_TRIALS", "100"))
    PHIAI_PARALLEL_JOBS: int = int(os.getenv("PHIAI_PARALLEL_JOBS", "1"))
    PHIAI_WALK_FORWARD_WINDOWS: int = int(os.getenv("PHIAI_WALK_FORWARD_WINDOWS", "3"))

    BROKER_API_KEY: str = os.getenv("BROKER_API_KEY", "")
    BROKER_SECRET_KEY: str = os.getenv("BROKER_SECRET_KEY", "")
    BROKER_BASE_URL: str = os.getenv("BROKER_BASE_URL", "https://paper-api.alpaca.markets")
    LIVE_MODE: str = os.getenv("LIVE_MODE", "paper")
    LIVE_SYMBOLS: tuple[str, ...] = _env_list("LIVE_SYMBOLS", ("SPY",))
    LIVE_UPDATE_INTERVAL: int = int(os.getenv("LIVE_UPDATE_INTERVAL", "60"))
    LIVE_CONFIG_PATH: Path = Path(os.getenv("LIVE_CONFIG_PATH", "./live_config.json"))

    @property
    def DATA_CACHE_ROOT(self) -> Path:
        """Backward-compatible alias for ``DATA_CACHE_DIR``."""
        return self.DATA_CACHE_DIR

    def create_dirs(self) -> None:
        """Create configured runtime directories if they do not exist."""
        for directory in (self.DATA_CACHE_DIR, self.RUNS_DIR, self.REGIME_MODELS_DIR, self.LOGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
