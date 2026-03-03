"""
phinance.utils.timing
=====================

High-resolution timers for profiling data fetches, backtests, and
optimization runs.

Usage
-----
    from phinance.utils.timing import Timer, timeit

    # Context-manager style
    with Timer("backtest") as t:
        run_direct_backtest(...)
    print(f"Took {t.elapsed_ms:.0f} ms")

    # Decorator style
    @timeit
    def expensive():
        ...
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """Context-manager wall-clock timer.

    Attributes
    ----------
    name : str
        Human-readable label for the timed block.
    elapsed_ms : float
        Wall-clock duration in milliseconds (set after ``__exit__``).
    elapsed_s : float
        Wall-clock duration in seconds (set after ``__exit__``).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_s = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed_s * 1_000

    def __repr__(self) -> str:
        if self.name:
            return f"Timer({self.name!r}, elapsed={self.elapsed_ms:.1f}ms)"
        return f"Timer(elapsed={self.elapsed_ms:.1f}ms)"


def timeit(func: F) -> F:
    """Decorator that prints elapsed time after every call.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function with identical signature.

    Example
    -------
        @timeit
        def my_slow_function():
            time.sleep(0.5)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(func.__qualname__) as t:
            result = func(*args, **kwargs)
        print(f"[timeit] {func.__qualname__} took {t.elapsed_ms:.1f} ms")
        return result

    return wrapper  # type: ignore[return-value]


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string.

    Examples
    --------
    >>> format_duration(0.045)
    '45.0 ms'
    >>> format_duration(3.7)
    '3.70 s'
    >>> format_duration(125)
    '2m 05s'
    """
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"
