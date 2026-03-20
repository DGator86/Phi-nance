"""Utilities for lightweight profiling and benchmark timing.

This module complements :mod:`phinance.utils.timing` with structured collection
of timing samples and optional cProfile capture for baseline investigations.
"""

from __future__ import annotations

import cProfile
import contextlib
import functools
import io
import pstats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingSample:
    """Single timing sample recorded for a named operation."""

    name: str
    seconds: float


class PerformanceTracker:
    """Collect and summarize timing samples for profiling runs."""

    def __init__(self) -> None:
        self._samples: List[TimingSample] = []

    def record(self, name: str, seconds: float) -> None:
        self._samples.append(TimingSample(name=name, seconds=seconds))

    @property
    def samples(self) -> List[TimingSample]:
        return list(self._samples)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return aggregate stats by sample name."""
        grouped: Dict[str, List[float]] = {}
        for sample in self._samples:
            grouped.setdefault(sample.name, []).append(sample.seconds)

        out: Dict[str, Dict[str, float]] = {}
        for name, values in grouped.items():
            total = sum(values)
            out[name] = {
                "count": float(len(values)),
                "total_seconds": total,
                "avg_seconds": total / len(values),
                "max_seconds": max(values),
                "min_seconds": min(values),
            }
        return out

    def as_markdown(self) -> str:
        """Render timing summary as a markdown table."""
        summary = self.summary()
        if not summary:
            return "_No timing samples recorded._"

        lines = [
            "| name | count | total (s) | avg (s) | min (s) | max (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, stats in sorted(summary.items()):
            lines.append(
                "| {name} | {count:.0f} | {total_seconds:.6f} | {avg_seconds:.6f} | {min_seconds:.6f} | {max_seconds:.6f} |".format(
                    name=name,
                    **stats,
                )
            )
        return "\n".join(lines)


@contextlib.contextmanager
def track_time(
    tracker: PerformanceTracker,
    name: str,
) -> Generator[None, None, None]:
    """Context manager to measure block duration and store it on ``tracker``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        tracker.record(name, time.perf_counter() - start)


def profiled(
    tracker: PerformanceTracker,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to measure function runtime and push a sample into tracker."""

    def decorator(func: F) -> F:
        sample_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with track_time(tracker, sample_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def run_cprofile(
    fn: Callable[..., Any],
    *args: Any,
    sort_by: str = "cumtime",
    top_n: int = 30,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Execute ``fn`` under cProfile and return formatted stats text."""
    profiler = cProfile.Profile()
    profiler.enable()
    fn(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_by)
    stats.print_stats(top_n)
    text = stream.getvalue()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return text
