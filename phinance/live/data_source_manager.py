"""Rate-limit aware multi-source data manager for live trading."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from phinance.features.extractor import FeatureExtractor
from phinance.live.cache import PersistentCache
from phinance.live.rate_limiter import RateLimiter
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

Fetcher = Callable[..., Any]


@dataclass
class SourceStatus:
    enabled: bool = True
    health: str = "ok"
    last_error: str | None = None
    last_success_at: str | None = None


class DataSourceManager:
    """Coordinates source priority, caching, rate limits, and usage tracking."""

    def __init__(self, config: dict[str, Any], cache: PersistentCache | None = None) -> None:
        self.config = config
        self.cache = cache or PersistentCache()
        self.sources: dict[str, dict[str, Fetcher]] = defaultdict(dict)
        self.limiters: dict[str, RateLimiter] = {}
        self.usage: dict[str, int] = defaultdict(int)
        self.usage_daily: dict[str, int] = defaultdict(int)
        self.daily_limit: dict[str, int] = {}
        self.status: dict[str, SourceStatus] = {}
        self.priorities: dict[str, list[str]] = config.get("data_priorities", {})
        self.cache_ttl: dict[str, float] = config.get("cache_ttl_seconds", {})
        self.feature_registry_path: Path | None = None
        self.feature_window = int(config.get("feature_window", 32))
        self.feature_extractor: FeatureExtractor | None = None

        feature_registry_path = config.get("feature_registry_path")
        if feature_registry_path:
            self.feature_registry_path = Path(feature_registry_path)
            if self.feature_registry_path.exists():
                self.feature_extractor = FeatureExtractor(
                    registry_path=self.feature_registry_path,
                    use_autoencoder=bool(config.get("use_auto_features", True)),
                    use_gp_features=bool(config.get("use_gp_features", True)),
                    window=self.feature_window,
                )

        self._init_sources(config.get("data_sources", {}))

    def _init_sources(self, source_cfg: dict[str, Any]) -> None:
        for source, cfg in source_cfg.items():
            enabled = bool(cfg.get("enabled", True))
            self.status[source] = SourceStatus(enabled=enabled)
            rate = cfg.get("rate_limit")
            per = cfg.get("rate_window_seconds", 60)
            if rate:
                self.limiters[source] = RateLimiter(rate=float(rate), per=float(per))
            if cfg.get("daily_limit"):
                self.daily_limit[source] = int(cfg["daily_limit"])

    def register_source(self, source: str, data_type: str, fetcher: Fetcher) -> None:
        self.sources[source][data_type] = fetcher

    def set_source_enabled(self, source: str, enabled: bool) -> None:
        if source not in self.status:
            self.status[source] = SourceStatus(enabled=enabled)
        self.status[source].enabled = enabled

    def _cache_key(self, source: str, data_type: str, **params: Any) -> str:
        parts = [source, data_type]
        for key in sorted(params.keys()):
            parts.append(f"{key}={params[key]}")
        return "|".join(parts)

    def _can_use_source(self, source: str) -> bool:
        if not self.status.get(source, SourceStatus()).enabled:
            return False
        if source in self.daily_limit and self.usage_daily[source] >= self.daily_limit[source]:
            self._mark_failure(source, f"daily limit reached ({self.daily_limit[source]})")
            return False
        return True

    def _acquire_rate(self, source: str) -> bool:
        limiter = self.limiters.get(source)
        if limiter is None:
            return True
        decision = limiter.acquire(blocking=False)
        return decision.allowed

    def fetch(self, data_type: str, *, cache_params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        priority_sources = self.priorities.get(data_type, [])
        cache_params = cache_params or kwargs

        last_error: Exception | None = None
        for source in priority_sources:
            if source not in self.sources or data_type not in self.sources[source]:
                continue
            if not self._can_use_source(source):
                continue

            cache_key = self._cache_key(source, data_type, **cache_params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            if not self._acquire_rate(source):
                logger.warning("Skipping %s for %s due to rate limit", source, data_type)
                continue

            try:
                payload = self.sources[source][data_type](**kwargs)
                ttl = self.cache_ttl.get(data_type, 300)
                self.cache.set(cache_key, payload, expiry_seconds=ttl)
                self.usage[source] += 1
                self.usage_daily[source] += 1
                self._mark_success(source)
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._mark_failure(source, str(exc))
                logger.warning("Data fetch failed from %s for %s: %s", source, data_type, exc)
                continue

        if last_error:
            raise RuntimeError(f"All sources exhausted for {data_type}: {last_error}")
        raise RuntimeError(f"All sources exhausted for {data_type}")

    def usage_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for source, status in self.status.items():
            limit = self.daily_limit.get(source)
            used = self.usage_daily.get(source, 0)
            remaining = None if limit is None else max(0, limit - used)
            snapshot[source] = {
                "enabled": status.enabled,
                "health": status.health,
                "last_error": status.last_error,
                "last_success_at": status.last_success_at,
                "calls_made": self.usage.get(source, 0),
                "daily_used": used,
                "daily_limit": limit,
                "daily_remaining": remaining,
            }
        return snapshot

    def build_discovered_features(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Compute discovered features using configured registry, if available."""
        if self.feature_extractor is None:
            return {"features": [], "dim": 0, "enabled": False}
        values = self.feature_extractor.extract(market_data)
        return {
            "features": values.tolist(),
            "dim": int(values.shape[0]),
            "enabled": True,
        }

    def _mark_success(self, source: str) -> None:
        self.status[source].health = "ok"
        self.status[source].last_error = None
        self.status[source].last_success_at = datetime.now(timezone.utc).isoformat()

    def _mark_failure(self, source: str, error: str) -> None:
        self.status[source].health = "degraded"
        self.status[source].last_error = error
