"""Rate-limiting primitives for live data sources."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitDecision:
    """Decision metadata returned by :meth:`RateLimiter.acquire`."""

    allowed: bool
    wait_seconds: float = 0.0


class RateLimiter:
    """Thread-safe token bucket limiter.

    Parameters
    ----------
    rate:
        Maximum number of tokens available per ``per`` seconds.
    per:
        Window size in seconds.
    """

    def __init__(self, rate: float, per: float = 60.0) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if per <= 0:
            raise ValueError("per must be > 0")
        self.rate = float(rate)
        self.per = float(per)
        self.tokens = float(rate)
        self.last = time.monotonic()
        self.lock = Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self.last
        if elapsed <= 0:
            return
        self.tokens += elapsed * (self.rate / self.per)
        if self.tokens > self.rate:
            self.tokens = self.rate
        self.last = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> RateLimitDecision:
        """Acquire tokens.

        Returns a :class:`RateLimitDecision`; when ``blocking=False`` and the
        request would exceed the limit, the call returns immediately.
        """
        if tokens <= 0:
            return RateLimitDecision(allowed=True, wait_seconds=0.0)

        with self.lock:
            now = time.monotonic()
            self._refill(now)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitDecision(allowed=True, wait_seconds=0.0)

            need = tokens - self.tokens
            wait = need * (self.per / self.rate)
            if not blocking:
                return RateLimitDecision(allowed=False, wait_seconds=max(0.0, wait))

        time.sleep(max(0.0, wait))

        with self.lock:
            now = time.monotonic()
            self._refill(now)
            if self.tokens >= tokens:
                self.tokens -= tokens
            else:
                self.tokens = 0.0
                self.last = now
            return RateLimitDecision(allowed=True, wait_seconds=max(0.0, wait))
