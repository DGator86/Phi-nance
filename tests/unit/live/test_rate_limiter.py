from __future__ import annotations

from unittest.mock import patch

from phinance.live.rate_limiter import RateLimiter


def test_non_blocking_rejects_when_no_tokens():
    limiter = RateLimiter(rate=1, per=60)
    first = limiter.acquire(blocking=False)
    second = limiter.acquire(blocking=False)
    assert first.allowed is True
    assert second.allowed is False
    assert second.wait_seconds > 0


def test_blocking_sleeps_and_allows():
    limiter = RateLimiter(rate=1, per=60)
    limiter.acquire(blocking=False)

    with patch("phinance.live.rate_limiter.time.sleep") as sleep_mock:
        decision = limiter.acquire(blocking=True)

    assert decision.allowed is True
    sleep_mock.assert_called_once()
