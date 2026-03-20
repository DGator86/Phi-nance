"""Hyperparameter sweep runner for experiments."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from phinance.experiment.runner import run_experiment
from phinance.experiment.search_space import expand_search_space, generate_trial_overrides
from phinance.experiment.tracker import create_tracker

logger = logging.getLogger(__name__)


class SweepRunner:
    """Execute a sweep config by launching one run per sampled trial."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.raw_config: dict[str, Any] = yaml.safe_load(handle) or {}

    def _write_trial_config(self, trial_config: dict[str, Any], trial_index: int) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_trial_{trial_index}.yaml", delete=False, encoding="utf-8"
        )
        try:
            yaml.safe_dump(trial_config, handle)
            handle.flush()
        finally:
            handle.close()
        return Path(handle.name)

    def _build_trial_configs(self, parent_run_id: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        trial_overrides = generate_trial_overrides(self.raw_config)
        expanded_trials = expand_search_space(self.raw_config)

        for i, (trial_cfg, trial_override) in enumerate(zip(expanded_trials, trial_overrides), start=1):
            tags = dict(trial_cfg.get("tags", {}))
            tags.update(
                {
                    "sweep.name": str(self.raw_config.get("name", "sweep")),
                    "sweep.trial_index": str(i),
                }
            )
            if parent_run_id:
                tags["mlflow.parentRunId"] = parent_run_id
            trial_cfg["tags"] = tags
            tracking = dict(trial_cfg.get("tracking", {}))
            if parent_run_id and str(tracking.get("backend", "")).lower() == "mlflow":
                tracking["nested"] = True
                tracking["parent_run_id"] = parent_run_id
            trial_cfg["tracking"] = tracking
            trial_cfg["description"] = (
                f"{self.raw_config.get('description', '')} | trial {i} overrides={trial_override}"
            ).strip()

        return expanded_trials, trial_overrides

    def _run_trial(self, trial_config: dict[str, Any], trial_index: int) -> dict[str, Any]:
        cfg_path = self._write_trial_config(trial_config, trial_index)
        try:
            result = run_experiment(str(cfg_path))
            return {
                "trial_index": trial_index,
                "status": result.get("status", "UNKNOWN"),
                "run_id": result.get("run_id"),
                "run_url": result.get("run_url"),
                "metrics": result.get("metrics", {}),
            }
        except Exception as exc:  # pragma: no cover - error path validated indirectly
            logger.exception("Sweep trial %s failed", trial_index)
            return {
                "trial_index": trial_index,
                "status": "FAILED",
                "error": str(exc),
                "metrics": {},
            }

    def _write_trials_csv(self, trial_rows: list[dict[str, Any]]) -> Path:
        csv_path = Path(tempfile.NamedTemporaryFile(suffix="_sweep_trials.csv", delete=False).name)
        metric_keys: set[str] = set()
        for row in trial_rows:
            metric_keys.update((row.get("metrics") or {}).keys())
        fieldnames = ["trial_index", "status", "run_id", "run_url", "error", *sorted(metric_keys)]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in trial_rows:
                metrics = row.get("metrics") or {}
                output = {
                    "trial_index": row.get("trial_index"),
                    "status": row.get("status"),
                    "run_id": row.get("run_id"),
                    "run_url": row.get("run_url"),
                    "error": row.get("error", ""),
                }
                output.update(metrics)
                writer.writerow(output)
        return csv_path

    def run(self) -> dict[str, Any]:
        tracking_cfg = dict(self.raw_config.get("tracking", {}))
        backend = str(tracking_cfg.pop("backend", "none"))
        tracking_uri = tracking_cfg.pop("uri", None)

        parent_tracker = create_tracker(
            backend,
            experiment_name=str(self.raw_config.get("name", "sweep")),
            tracking_uri=tracking_uri,
            run_name=f"{self.raw_config.get('name', 'sweep')}_parent",
            tags={"kind": "sweep_parent", **{k: str(v) for k, v in self.raw_config.get("tags", {}).items()}},
            **tracking_cfg,
        )

        parent_status = "FINISHED"
        try:
            parent_run_id = parent_tracker.get_run_id()
            parent_tracker.log_params(
                {
                    "sweep.method": str(self.raw_config.get("sweep", {}).get("method", "grid")),
                    "sweep.parallel": int(self.raw_config.get("sweep", {}).get("parallel", 1)),
                    "sweep.n_trials": int(self.raw_config.get("sweep", {}).get("n_trials", 0)),
                    "target": str(self.raw_config.get("target", "")),
                }
            )
            config_snapshot = Path(tempfile.NamedTemporaryFile(suffix="_sweep_config.yaml", delete=False).name)
            config_snapshot.write_text(yaml.safe_dump(self.raw_config), encoding="utf-8")
            parent_tracker.log_artifact(str(config_snapshot))

            trial_configs, trial_overrides = self._build_trial_configs(parent_run_id=parent_run_id)
            parallelism = int(self.raw_config.get("sweep", {}).get("parallel", 1))

            trial_rows: list[dict[str, Any]] = []
            if parallelism <= 1:
                for i, cfg in enumerate(trial_configs, start=1):
                    row = self._run_trial(cfg, i)
                    row["overrides"] = trial_overrides[i - 1]
                    trial_rows.append(row)
            else:
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {
                        executor.submit(self._run_trial, cfg, i): i
                        for i, cfg in enumerate(trial_configs, start=1)
                    }
                    for future in as_completed(futures):
                        trial_index = futures[future]
                        row = future.result()
                        row["overrides"] = trial_overrides[trial_index - 1]
                        trial_rows.append(row)
                trial_rows.sort(key=lambda row: row["trial_index"])

            completed = [row for row in trial_rows if row.get("status") == "FINISHED"]
            failed = [row for row in trial_rows if row.get("status") != "FINISHED"]

            parent_tracker.log_metrics(
                {
                    "sweep.total_trials": float(len(trial_rows)),
                    "sweep.completed_trials": float(len(completed)),
                    "sweep.failed_trials": float(len(failed)),
                }
            )

            objective = str(self.raw_config.get("sweep", {}).get("objective_metric", "")).strip()
            best_trial = None
            if objective:
                candidates = [row for row in completed if objective in (row.get("metrics") or {})]
                if candidates:
                    mode = str(self.raw_config.get("sweep", {}).get("objective_mode", "max")).lower()
                    reverse = mode != "min"
                    best_trial = sorted(
                        candidates,
                        key=lambda row: float(row["metrics"][objective]),
                        reverse=reverse,
                    )[0]
                    parent_tracker.log_metrics({"sweep.best_objective": float(best_trial["metrics"][objective])})
                    parent_tracker.set_tags(
                        {
                            "sweep.best_trial_index": str(best_trial["trial_index"]),
                            "sweep.objective_metric": objective,
                            "sweep.objective_mode": mode,
                            "sweep.best_overrides": json.dumps(best_trial.get("overrides", {}), sort_keys=True),
                        }
                    )

            trials_csv = self._write_trials_csv(trial_rows)
            parent_tracker.log_artifact(str(trials_csv))

            return {
                "status": parent_status,
                "sweep_run_id": parent_run_id,
                "total_trials": len(trial_rows),
                "completed_trials": len(completed),
                "failed_trials": len(failed),
                "best_trial": best_trial,
                "trials": trial_rows,
            }
        except Exception:
            parent_status = "FAILED"
            raise
        finally:
            parent_tracker.finish(status=parent_status)
