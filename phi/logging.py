"""Centralized logging utilities for the ``phi`` package."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from phi.config import settings


_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _resolve_level(log_level: Optional[str]) -> int:
    level_name = (log_level or settings.LOG_LEVEL or "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging(
    name: str = "phi",
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with consistent handlers/formatting."""
    logger = logging.getLogger(name)
    level = _resolve_level(log_level)
    logger.setLevel(level)

    formatter = logging.Formatter(_FORMAT)
    resolved_file = log_file or (settings.LOGS_DIR / "phi.log")

    existing_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if console and not existing_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if resolved_file is not None:
        resolved_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(resolved_file.resolve())
        existing_file = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved_path
            for h in logger.handlers
        )
        if not existing_file:
            file_handler = logging.FileHandler(resolved_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a namespaced application logger for a module."""
    if not module_name.startswith("phi"):
        module_name = f"phi.{module_name}"
    return logging.getLogger(module_name)
