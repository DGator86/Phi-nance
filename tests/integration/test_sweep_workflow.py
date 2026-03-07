from __future__ import annotations

import yaml


def test_sweep_workflow_with_mlflow(tmp_path):
    mlflow = __import__("pytest").importorskip("mlflow")

    from phinance.experiment.sweep import SweepRunner

    tracking_dir = tmp_path / "mlruns"
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "integration_sweep",
                "tracking": {"backend": "mlflow", "uri": str(tracking_dir)},
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "params": {"generations": 1},
                "search_space": {
                    "params.generations": {"type": "choice", "values": [1, 2, 3]},
                },
                "sweep": {
                    "method": "grid",
                    "parallel": 1,
                    "objective_metric": "best_score",
                    "objective_mode": "max",
                },
            }
        ),
        encoding="utf-8",
    )

    result = SweepRunner(cfg_path).run()
    assert result["status"] == "FINISHED"
    assert result["total_trials"] == 3
    assert result["completed_trials"] == 3

    mlflow.set_tracking_uri(str(tracking_dir))
    exp = mlflow.get_experiment_by_name("integration_sweep")
    assert exp is not None
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) >= 4  # 1 parent + 3 trials
