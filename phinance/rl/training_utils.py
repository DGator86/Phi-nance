"""Utility helpers for opt-in RL training performance optimisation."""

from __future__ import annotations

import contextlib
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]


@dataclass
class RuntimeConfig:
    use_gpu: bool = False
    mixed_precision: bool = False


class NullGradScaler:
    """CPU fallback with GradScaler-like interface."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


@contextlib.contextmanager
def autocast_context(enabled: bool, device_type: str) -> Iterator[None]:
    if enabled and device_type == "cuda":
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            yield
    else:
        yield


def resolve_device(runtime_cfg: RuntimeConfig) -> torch.device:
    if runtime_cfg.use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_grad_scaler(runtime_cfg: RuntimeConfig) -> NullGradScaler | torch.cuda.amp.GradScaler:
    if runtime_cfg.use_gpu and runtime_cfg.mixed_precision and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return NullGradScaler()


def move_policy_to_device(policy: torch.nn.Module, runtime_cfg: RuntimeConfig) -> Tuple[torch.nn.Module, torch.device]:
    device = resolve_device(runtime_cfg)
    policy.to(device)
    return policy, device


def state_cache(maxsize: int = 4096) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator for caching repeated deterministic state computations."""

    def decorator(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        @functools.lru_cache(maxsize=maxsize)
        def _cached(key: Tuple[float, ...]) -> Tuple[float, ...]:
            arr = np.asarray(key, dtype=np.float64)
            out = fn(arr)
            return tuple(float(x) for x in out)

        @functools.wraps(fn)
        def wrapper(state: np.ndarray) -> np.ndarray:
            rounded = tuple(np.round(np.asarray(state, dtype=np.float64), 6))
            return np.asarray(_cached(rounded), dtype=np.float32)

        return wrapper

    return decorator


def load_optimisation_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"rl_optimisation": {}}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "rl_optimisation" not in data:
        data = {"rl_optimisation": data}
    return data


def get_runtime_config(optim_cfg: Dict[str, Any]) -> RuntimeConfig:
    gpu_cfg = optim_cfg.get("rl_optimisation", {}).get("gpu", {})
    return RuntimeConfig(
        use_gpu=bool(gpu_cfg.get("enabled", False)),
        mixed_precision=bool(gpu_cfg.get("mixed_precision", False)),
    )


def python_step_kernel(
    remaining_shares: float,
    fraction: float,
    volume: float,
    participation_cap: float,
    spread: float,
    mid: float,
    arrival_price: float,
    impact_coefficient: float,
    urgency: float,
    urgency_cross_spread: float,
    adv: float,
    total_shares: float,
) -> Tuple[float, float, float, float, float]:
    desired_shares = remaining_shares * fraction
    max_slice = participation_cap * max(volume, 1.0)
    executed_shares = min(remaining_shares, desired_shares, max_slice)
    impact = impact_coefficient * (executed_shares / max(adv, 1.0)) * spread
    urgency_penalty = urgency * urgency_cross_spread * spread
    execution_price = mid + impact + urgency_penalty
    slippage = (execution_price - arrival_price) / max(arrival_price, 1e-6)
    reward = -slippage * (executed_shares / max(total_shares, 1.0))
    return executed_shares, execution_price, slippage, reward, impact


if njit is not None:  # pragma: no cover - numba execution environment-dependent
    step_kernel = njit(cache=True)(python_step_kernel)
else:
    step_kernel = python_step_kernel
    logger.info("Numba not available, using python step kernel fallback.")
