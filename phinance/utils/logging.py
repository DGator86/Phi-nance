"""
phinance.utils.logging
======================

Centralised logging configuration for the Phi-nance package.

Usage
-----
    from phinance.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Fetching %s bars for %s", n, symbol)

All loggers inherit from the ``phinance`` root logger so callers can
configure the entire library with a single ``logging.getLogger("phinance")``
call.

Default format: ``[LEVEL] phinance.module: message``
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_ROOT_LOGGER_NAME = "phinance"
_DEFAULT_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
_INITIALISED = False


def _init_root_logger(level: int = logging.WARNING) -> None:
    """Attach a StreamHandler to the root phinance logger if none exist yet.

    This is a one-time initialisation so library code never silently swallows
    messages, but does not force a specific handler on the calling application.
    """
    global _INITIALISED
    if _INITIALISED:
        return
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)
    _INITIALISED = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger beneath the ``phinance`` root.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.  If it does not already
        start with ``"phinance"`` the prefix is added automatically so all
        library loggers share the same hierarchy.
    level : int, optional
        Override the level on this specific logger (default: inherits root).

    Returns
    -------
    logging.Logger
    """
    _init_root_logger()
    # Normalise name to phinance.* hierarchy
    if not name.startswith(_ROOT_LOGGER_NAME):
        # e.g. "__main__" → "phinance.__main__"
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Convenience helper for applications that use Phi-nance as a library.

    Sets the root ``phinance`` logger level and (optionally) replaces the
    default formatter.  Call this once at application start-up:

        from phinance.utils.logging import configure_logging
        configure_logging(logging.DEBUG)

    Parameters
    ----------
    level : int
        Desired log level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    fmt : str, optional
        Custom format string.  Defaults to ``_DEFAULT_FORMAT``.
    """
    global _INITIALISED
    _INITIALISED = False  # Allow re-initialisation with new settings
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    # Remove existing handlers before re-adding
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    _INITIALISED = True
