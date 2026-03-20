"""Unified experiment runner."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from phinance.experiment.config_schema import apply_overrides, load_experiment_config, validate_experiment_config
from phinance.experiment.tracker import create_tracker

logger = logging.getLogger(__name__)

TARGET_REGISTRY: dict[str, str] = {
    "train_execution_agent": "scripts.train_execution_agent:run_experiment_target",
    "train_strategy_rd_agent": "scripts.train_strategy_rd_agent:run_experiment_target",
    "train_risk_monitor_agent": "scripts.train_risk_monitor_agent:run_experiment_target",
    "train_meta_agent": "scripts.train_meta_agent:run_experiment_target",
    "run_gp_search": "scripts.run_gp_search:run_experiment_target",
    "run_backtest": "scripts.run_backtest:run_experiment_target",
}


def _resolve_target(target: str) -> Callable[..., dict[str, float]]:
    spec = TARGET_REGISTRY.get(target, target)
    if ":" not in spec:
        raise ValueError(f"Invalid target spec: {target}. Expected module:function")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Target function is not callable: {spec}")
    return fn


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        code = subprocess.call(["git", "diff", "--quiet"])
        return code != 0
    except Exception:
        return None


def _write_environment_snapshot() -> Path:
    requirements = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_requirements.txt", delete=False, encoding="utf-8")
    try:
        tmp.write(requirements)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def run_experiment(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = apply_overrides(raw, overrides)
    validate_experiment_config(raw)

    config = load_experiment_config(path)
    if overrides:
        config = config.__class__(
            name=str(raw["name"]),
            description=raw.get("description"),
            tracking=dict(raw.get("tracking", {})),
            target=str(raw["target"]),
            params=dict(raw.get("params", {})),
            data=dict(raw.get("data", {})),
            tags={k: str(v) for k, v in dict(raw.get("tags", {})).items()},
        )

    tracking_cfg = dict(config.tracking)
    backend = str(tracking_cfg.pop("backend", "none"))
    tracking_uri = tracking_cfg.pop("uri", None)

    tracker = create_tracker(
        backend,
        experiment_name=config.name,
        tracking_uri=tracking_uri,
        run_name=config.name,
        tags=config.tags,
        **tracking_cfg,
    )

    snapshot_path = _write_environment_snapshot()
    params_to_log = {**config.params, **{f"data.{k}": v for k, v in config.data.items()}}
    tracker.log_params(params_to_log)

    run_tags = dict(config.tags)
    commit_hash = _git_commit_hash()
    if commit_hash:
        run_tags["git_commit"] = commit_hash
    dirty = _git_dirty()
    if dirty is not None:
        run_tags["git_dirty"] = str(dirty)
    if config.description:
        run_tags["description"] = config.description
    tracker.set_tags(run_tags)
    tracker.log_artifact(str(snapshot_path))

    target_fn = _resolve_target(config.target)
    logger.info("Running experiment target %s", config.target)
    status = "FINISHED"
    try:
        metrics = target_fn(**config.params, tracker=tracker)
        if metrics:
            tracker.log_metrics({k: float(v) for k, v in metrics.items()})
        result = {
            "run_id": tracker.get_run_id(),
            "run_url": tracker.get_run_url(),
            "metrics": metrics or {},
            "status": status,
        }
        return result
    except Exception:
        status = "FAILED"
        tracker.set_tags({"failure": "true"})
        raise
    finally:
        tracker.finish(status=status)
