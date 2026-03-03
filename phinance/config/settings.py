"""
phinance.config.settings
=========================

Global application settings loaded from environment variables / .env file.

Settings are loaded once at import time via ``get_settings()`` and cached.
Applications can override them at runtime by modifying ``Settings`` attributes.

Environment variables
---------------------
  AV_API_KEY              — Alpha Vantage API key
  MARKETDATAAPP_API_TOKEN — MarketDataApp token for options data
  OLLAMA_HOST             — Ollama server URL (default: http://localhost:11434)
  OLLAMA_MODEL            — Default Ollama model (default: llama3.2)
  PHINANCE_DATA_DIR       — Override data_cache root
  PHINANCE_RUNS_DIR       — Override runs root
  PHINANCE_LOG_LEVEL      — Logging level (default: WARNING)

Usage
-----
    from phinance.config.settings import get_settings
    s = get_settings()
    print(s.av_api_key)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Try to load .env automatically when python-dotenv is available
try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent.parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass


class Settings:
    """Application-level settings resolved from environment variables.

    Attributes
    ----------
    av_api_key              : Alpha Vantage API key (or empty string)
    marketdataapp_api_token : MarketDataApp token (or empty string)
    ollama_host             : Ollama server base URL
    ollama_model            : Default Ollama model name
    data_cache_dir          : Root for OHLCV parquet cache
    runs_dir                : Root for backtest run storage
    log_level               : Python logging level integer
    """

    def __init__(self) -> None:
        _root = Path(__file__).resolve().parent.parent.parent

        self.av_api_key: str = os.environ.get("AV_API_KEY", "")
        self.marketdataapp_api_token: str = os.environ.get(
            "MARKETDATAAPP_API_TOKEN", ""
        )
        self.ollama_host: str = os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        self.ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")

        self.data_cache_dir: Path = Path(
            os.environ.get("PHINANCE_DATA_DIR", str(_root / "data_cache"))
        )
        self.runs_dir: Path = Path(
            os.environ.get("PHINANCE_RUNS_DIR", str(_root / "runs"))
        )
        self.log_level: int = getattr(
            logging,
            os.environ.get("PHINANCE_LOG_LEVEL", "WARNING").upper(),
            logging.WARNING,
        )

    def __repr__(self) -> str:
        masked_av = f"{self.av_api_key[:4]}****" if self.av_api_key else "(not set)"
        return (
            f"Settings(av_api_key={masked_av}, "
            f"ollama={self.ollama_host}, "
            f"data_cache={self.data_cache_dir})"
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton (lazy-initialised)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
