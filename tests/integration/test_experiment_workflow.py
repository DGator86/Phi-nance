from __future__ import annotations

import yaml

from phinance.experiment.runner import run_experiment


def test_experiment_workflow_end_to_end(tmp_path):
    cfg_path = tmp_path / "workflow.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "workflow",
                "description": "integration smoke",
                "tracking": {"backend": "none"},
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "params": {"generations": 3},
                "data": {"symbols": ["SPY"]},
                "tags": {"kind": "integration"},
            }
        )
    )

    result = run_experiment(str(cfg_path), overrides=["params.generations=4"])
    assert result["status"] == "FINISHED"
    assert result["metrics"]["best_score"] == 4.0
