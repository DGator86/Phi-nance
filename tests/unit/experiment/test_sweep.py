from __future__ import annotations

import yaml

from phinance.experiment.sweep import SweepRunner


def test_sweep_runner_executes_all_trials(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "unit_sweep",
                "tracking": {"backend": "none"},
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "params": {"generations": 1},
                "search_space": {
                    "params.generations": {"type": "choice", "values": [1, 2, 3]},
                },
                "sweep": {"method": "grid", "parallel": 1},
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_experiment(config_path: str, overrides=None):
        with open(config_path, "r", encoding="utf-8") as handle:
            trial_cfg = yaml.safe_load(handle)
        calls.append(trial_cfg["params"]["generations"])
        return {
            "status": "FINISHED",
            "run_id": f"run-{trial_cfg['params']['generations']}",
            "run_url": None,
            "metrics": {"best_score": float(trial_cfg["params"]["generations"])},
        }

    monkeypatch.setattr("phinance.experiment.sweep.run_experiment", fake_run_experiment)

    result = SweepRunner(cfg_path).run()
    assert result["status"] == "FINISHED"
    assert result["total_trials"] == 3
    assert result["failed_trials"] == 0
    assert sorted(calls) == [1, 2, 3]


def test_sweep_runner_single_run_compatibility(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "single.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "single_point",
                "tracking": {"backend": "none"},
                "target": "phinance.experiment.dummy_targets:gp_discovery_target",
                "params": {"generations": 5},
            }
        ),
        encoding="utf-8",
    )

    call_count = {"n": 0}

    def fake_run_experiment(config_path: str, overrides=None):
        call_count["n"] += 1
        return {"status": "FINISHED", "run_id": "run-single", "run_url": None, "metrics": {}}

    monkeypatch.setattr("phinance.experiment.sweep.run_experiment", fake_run_experiment)

    result = SweepRunner(cfg_path).run()
    assert result["total_trials"] == 1
    assert call_count["n"] == 1
