"""Centralized logging utilities for the ``phi`` package."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional


_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line for log aggregators (Docker / Loki / CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _use_json_logs() -> bool:
    return os.environ.get("PHINANCE_LOG_JSON", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _resolve_level(log_level: Optional[str]) -> int:
    if log_level:
        level_name = log_level.upper()
    else:
        from phi.config import settings

        level_name = (settings.LOG_LEVEL or "INFO").upper()
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

    formatter: logging.Formatter
    if _use_json_logs():
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_FORMAT)
    if log_file is None:
        from phi.config import settings

        resolved_file = settings.LOGS_DIR / "phi.log"
    else:
        resolved_file = log_file

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
