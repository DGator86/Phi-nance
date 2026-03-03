"""
phinance.utils.decorators
=========================

Reusable function decorators for the Phi-nance package.

Decorators
----------
retry               — Exponential-backoff retry with configurable exceptions.
timeit_decorator    — Same as ``timeit`` from timing.py; re-exported for
                      convenience so callers import from one place.
log_call            — Logs function entry/exit with arguments.
validate_ohlcv      — Assert first positional arg is a valid OHLCV DataFrame.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_logger = logging.getLogger("phinance.utils.decorators")


# ── retry ────────────────────────────────────────────────────────────────────


def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """Exponential-backoff retry decorator.

    Parameters
    ----------
    exceptions : tuple of Exception types
        Only retry on these exception types.
    max_attempts : int
        Maximum total attempts (including the first call).
    base_delay : float
        Seconds to wait before the second attempt.
    backoff : float
        Multiplier applied to delay after each failure.
    logger : logging.Logger, optional
        Logs warnings on each retry; defaults to phinance root logger.

    Example
    -------
        @retry(exceptions=(requests.Timeout,), max_attempts=3)
        def call_api():
            ...
    """
    log = logger or _logger

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: Exception = Exception("Unknown error")
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        log.warning(
                            "%s attempt %d/%d failed: %s. Retrying in %.1fs.",
                            func.__qualname__, attempt, max_attempts, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= backoff
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


# ── timeit_decorator ─────────────────────────────────────────────────────────


def timeit_decorator(func: F) -> F:
    """Print wall-clock time after every call.  Alias for ``timing.timeit``."""
    from phinance.utils.timing import timeit as _timeit
    return _timeit(func)


# ── log_call ─────────────────────────────────────────────────────────────────


def log_call(level: int = logging.DEBUG) -> Callable[[F], F]:
    """Log function entry and exit at the given level.

    Example
    -------
        @log_call(logging.INFO)
        def run(config):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger.log(level, "ENTER %s", func.__qualname__)
            try:
                result = func(*args, **kwargs)
                _logger.log(level, "EXIT  %s → OK", func.__qualname__)
                return result
            except Exception as exc:
                _logger.log(level, "EXIT  %s → ERROR: %s", func.__qualname__, exc)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# ── validate_ohlcv ───────────────────────────────────────────────────────────


def validate_ohlcv(min_rows: int = 2) -> Callable[[F], F]:
    """Assert that the first positional argument is a valid OHLCV DataFrame.

    Raises ``phinance.exceptions.InsufficientDataError`` when the check fails.

    Parameters
    ----------
    min_rows : int
        Minimum required row count.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import pandas as pd
            from phinance.exceptions import InsufficientDataError

            if not args:
                return func(*args, **kwargs)
            df = args[0]
            if not isinstance(df, pd.DataFrame):
                return func(*args, **kwargs)
            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise InsufficientDataError(
                    f"OHLCV DataFrame missing columns: {sorted(missing)}"
                )
            if len(df) < min_rows:
                raise InsufficientDataError(
                    f"Need at least {min_rows} rows; got {len(df)}"
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
