"""Experiment config validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    description: str | None
    tracking: dict[str, Any]
    target: str
    params: dict[str, Any]
    data: dict[str, Any]
    tags: dict[str, str]


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    validate_experiment_config(raw)
    return ExperimentConfig(
        name=str(raw["name"]),
        description=raw.get("description"),
        tracking=dict(raw.get("tracking", {})),
        target=str(raw["target"]),
        params=dict(raw.get("params", {})),
        data=dict(raw.get("data", {})),
        tags={k: str(v) for k, v in dict(raw.get("tags", {})).items()},
    )


def validate_experiment_config(raw: dict[str, Any]) -> None:
    required = ["name", "target"]
    for key in required:
        if key not in raw or raw[key] in (None, ""):
            raise ValueError(f"Experiment config is missing required field: {key}")

    for section in ["tracking", "params", "data", "tags"]:
        if section in raw and not isinstance(raw[section], dict):
            raise TypeError(f"Section `{section}` must be a mapping/object")

    tracking = raw.get("tracking", {})
    backend = str(tracking.get("backend", "none")).lower()
    if backend not in {"none", "noop", "disabled", "mlflow", "wandb", "weights_biases", "weights-and-biases"}:
        raise ValueError(f"Unsupported tracking backend: {backend}")


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    data = dict(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key_path, value = item.split("=", 1)
        cursor = data
        keys = key_path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot apply override on non-object path: {key_path}")
        parsed: Any = value
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    pass
        cursor[keys[-1]] = parsed
    return data
