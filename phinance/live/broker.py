"""Broker facade used by the new live engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phinance.live.alpaca import AlpacaBroker
from phinance.live.rate_limiter import RateLimiter


@dataclass
class BrokerConfig:
    provider: str = "alpaca"
    rate_limit: float = 200.0
    rate_window_seconds: float = 60.0


class LiveBroker:
    """Thin broker wrapper adding generic call-throttling for broker APIs."""

    def __init__(self, broker: AlpacaBroker, config: BrokerConfig | None = None) -> None:
        self.broker = broker
        self.config = config or BrokerConfig()
        self.limiter = RateLimiter(rate=self.config.rate_limit, per=self.config.rate_window_seconds)

    def connect(self) -> None:
        self.broker.connect()

    def close(self) -> None:
        self.broker.close()

    def safe_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        self.limiter.acquire()
        method = getattr(self.broker, method_name)
        return method(*args, **kwargs)
