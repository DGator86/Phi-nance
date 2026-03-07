from __future__ import annotations

from pathlib import Path

import yaml

from phinance.experiment.config_schema import apply_overrides, load_experiment_config
from phinance.experiment.runner import run_experiment


def test_apply_overrides() -> None:
    raw = {"params": {"timesteps": 100}, "tracking": {"backend": "none"}}
    updated = apply_overrides(raw, ["params.timesteps=200", "params.flag=true"])
    assert updated["params"]["timesteps"] == 200
    assert updated["params"]["flag"] is True


def test_load_experiment_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "tracking": {"backend": "none"},
            }
        )
    )
    cfg = load_experiment_config(cfg_path)
    assert cfg.name == "demo"


def test_run_experiment_with_none_backend(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "tracking": {"backend": "none"},
                "params": {"generations": 2},
                "tags": {"suite": "unit"},
            }
        )
    )

    result = run_experiment(str(cfg_path))
    assert result["status"] == "FINISHED"
    assert result["metrics"]["best_score"] == 2.0
