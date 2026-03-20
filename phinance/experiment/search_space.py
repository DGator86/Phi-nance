"""Search space parsing and expansion utilities for experiment sweeps."""

from __future__ import annotations

import copy
from itertools import product
from typing import Any

import numpy as np


SUPPORTED_PARAM_TYPES = {"uniform", "loguniform", "int", "choice"}


def _validate_space(search_space: dict[str, dict[str, Any]]) -> None:
    for path, spec in search_space.items():
        if not isinstance(spec, dict):
            raise TypeError(f"Search space spec for `{path}` must be a mapping")
        param_type = str(spec.get("type", "")).lower()
        if param_type not in SUPPORTED_PARAM_TYPES:
            raise ValueError(f"Unsupported search space type `{param_type}` for `{path}`")
        if param_type == "choice":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Choice search space for `{path}` requires non-empty `values`")
            continue
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            raise ValueError(f"Search space type `{param_type}` for `{path}` requires `low` and `high`")
        if float(high) < float(low):
            raise ValueError(f"Search space bounds invalid for `{path}`: high must be >= low")
        if param_type == "loguniform" and (float(low) <= 0 or float(high) <= 0):
            raise ValueError(f"Loguniform bounds for `{path}` must be > 0")


def _set_dotted(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = config
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        nxt = cursor.get(key)
        if nxt is None:
            nxt = {}
            cursor[key] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot assign dotted path `{dotted_path}` through non-dict key `{key}`")
        cursor = nxt
    cursor[keys[-1]] = value


def _grid_values_for_spec(spec: dict[str, Any]) -> list[Any]:
    param_type = str(spec["type"]).lower()
    if param_type == "choice":
        return list(spec["values"])
    if param_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        return list(range(low, high + 1))
    # For continuous distributions in grid mode, require explicit values.
    values = spec.get("values")
    if isinstance(values, list) and values:
        return values
    raise ValueError(
        f"Grid sweeps for `{param_type}` require explicit `values` list."
    )


def _sample_for_spec(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    param_type = str(spec["type"]).lower()
    if param_type == "choice":
        values = spec["values"]
        idx = int(rng.integers(0, len(values)))
        return values[idx]
    if param_type == "int":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    if param_type == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if param_type == "loguniform":
        return float(np.exp(rng.uniform(np.log(float(spec["low"])), np.log(float(spec["high"])))))
    raise ValueError(f"Unsupported search space type: {param_type}")


def generate_trial_overrides(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate trial-level dotted overrides from a sweep config."""
    search_space = config.get("search_space") or {}
    if not search_space:
        return [{}]
    if not isinstance(search_space, dict):
        raise TypeError("`search_space` must be a mapping/object")

    _validate_space(search_space)
    sweep_cfg = dict(config.get("sweep", {}))
    method = str(sweep_cfg.get("method", "grid")).lower()

    if method == "grid":
        keys = list(search_space.keys())
        value_lists = [_grid_values_for_spec(search_space[key]) for key in keys]
        combinations: list[dict[str, Any]] = []
        for values in product(*value_lists):
            combinations.append(dict(zip(keys, values)))
        return combinations

    if method == "random":
        n_trials = int(sweep_cfg.get("n_trials", 1))
        if n_trials <= 0:
            raise ValueError("`sweep.n_trials` must be > 0 for random sweeps")
        seed = sweep_cfg.get("seed")
        rng = np.random.default_rng(seed)
        trials: list[dict[str, Any]] = []
        for _ in range(n_trials):
            trial: dict[str, Any] = {}
            for dotted_path, spec in search_space.items():
                trial[dotted_path] = _sample_for_spec(spec, rng)
            trials.append(trial)
        return trials

    raise ValueError(f"Unsupported sweep method: {method}")


def expand_search_space(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fully materialized trial configs from a base sweep config."""
    trial_overrides = generate_trial_overrides(config)
    trials: list[dict[str, Any]] = []
    for override in trial_overrides:
        trial_config = copy.deepcopy(config)
        trial_config.pop("search_space", None)
        trial_config.pop("sweep", None)
        for dotted_path, value in override.items():
            _set_dotted(trial_config, dotted_path, value)
        trials.append(trial_config)
    return trials
