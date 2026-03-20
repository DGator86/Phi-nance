"""Ray-powered distributed execution for independent backtest jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from phinance.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover - ray may be absent
    ray = None


def _run_single_backtest_remote(config: Dict[str, Any]) -> Dict[str, Any]:
    return _run_single_backtest_local(config)


if ray is not None:  # pragma: no branch
    _run_single_backtest_remote = ray.remote(_run_single_backtest_remote)


def _run_single_backtest_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one backtest config and return serialisable metrics."""
    from phinance.backtest.runner import run_backtest
    from phinance.backtest.vectorized import run_vectorized_backtest

    engine = str(config.get("engine", "vectorized")).lower()
    ohlcv = config["ohlcv"]
    symbol = str(config.get("symbol", "UNKNOWN"))

    if engine == "vectorized":
        result = run_vectorized_backtest(
            ohlcv=ohlcv,
            signal=config["signal"],
            symbol=symbol,
            signal_threshold=float(config.get("signal_threshold", 0.1)),
            position_style=str(config.get("position_style", "long_short")),
            initial_capital=float(config.get("initial_capital", 100_000.0)),
            position_size=float(config.get("position_size", 1.0)),
            transaction_cost=float(config.get("transaction_cost", 0.0)),
        )
        payload = result.to_dict()
        payload["engine"] = "vectorized"
        return payload

    if engine == "event":
        result = run_backtest(
            ohlcv=ohlcv,
            symbol=symbol,
            indicators=config.get("indicators"),
            blend_weights=config.get("blend_weights"),
            blend_method=str(config.get("blend_method", "weighted_sum")),
            signal_threshold=float(config.get("signal_threshold", 0.15)),
            initial_capital=float(config.get("initial_capital", 100_000.0)),
            position_size_pct=float(config.get("position_size_pct", 0.95)),
            regime_probs=config.get("regime_probs"),
        )
        payload = result.to_dict()
        payload["engine"] = "event"
        return payload

    raise ValueError(f"Unsupported engine '{engine}'. Use 'vectorized' or 'event'.")


@dataclass
class DistributedBacktestRunner:
    """Run many independent backtests with Ray, with sequential fallback."""

    enabled: bool = True
    use_ray: bool = True
    num_cpus: Optional[int] = None
    address: Optional[str] = None
    local_mode: bool = False
    retries: int = 1
    timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        self._ray_enabled = bool(self.enabled and self.use_ray and ray is not None)
        self._owns_ray_runtime = False

        if not self._ray_enabled:
            if self.enabled and self.use_ray and ray is None:
                logger.warning("Ray unavailable; falling back to sequential backtest execution.")
            return

        if not ray.is_initialized():
            init_kwargs: Dict[str, Any] = {"ignore_reinit_error": True, "local_mode": self.local_mode}
            if self.num_cpus is not None:
                init_kwargs["num_cpus"] = int(self.num_cpus)
            if self.address:
                init_kwargs["address"] = self.address
            ray.init(**init_kwargs)
            self._owns_ray_runtime = True

    @property
    def is_distributed(self) -> bool:
        return self._ray_enabled

    def shutdown(self) -> None:
        if self._ray_enabled and self._owns_ray_runtime and ray is not None and ray.is_initialized():
            ray.shutdown()
            self._owns_ray_runtime = False

    def run_parallel(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not configs:
            return []

        if not self._ray_enabled:
            return [self._run_with_resilience(cfg) for cfg in configs]

        refs = [_run_single_backtest_remote.options(max_retries=max(self.retries, 0)).remote(cfg) for cfg in configs]
        try:
            values = ray.get(refs, timeout=self.timeout_s)
        except Exception as exc:
            logger.warning("Distributed execution failed, retrying sequentially: %s", exc)
            return [self._run_with_resilience(cfg) for cfg in configs]

        return [{"status": "ok", "result": v, "error": None} for v in values]

    def _run_with_resilience(self, config: Dict[str, Any]) -> Dict[str, Any]:
        attempts = max(1, self.retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                return {"status": "ok", "result": _run_single_backtest_local(config), "error": None}
            except Exception as exc:
                if attempt >= attempts:
                    return {"status": "error", "result": None, "error": str(exc)}
        return {"status": "error", "result": None, "error": "unknown"}
