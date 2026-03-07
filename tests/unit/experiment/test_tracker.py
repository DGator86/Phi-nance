from __future__ import annotations

import types

import pytest

from phinance.experiment.mlflow_tracker import MLflowTracker
from phinance.experiment.tracker import NoOpTracker, create_tracker


def test_create_tracker_noop() -> None:
    tracker = create_tracker("none")
    assert isinstance(tracker, NoOpTracker)
    assert tracker.get_run_id() == "noop"


def test_create_tracker_invalid_backend() -> None:
    with pytest.raises(ValueError):
        create_tracker("bogus")


def test_mlflow_tracker_with_mocked_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls = {"params": None, "metrics": None, "status": None}

    class DummyRunInfo:
        run_id = "run-123"
        experiment_id = "exp-1"

    class DummyRun:
        info = DummyRunInfo()

    dummy = types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=lambda run_name=None: DummyRun(),
        log_params=lambda p: calls.__setitem__("params", p),
        log_metrics=lambda m, step=None: calls.__setitem__("metrics", (m, step)),
        log_artifact=lambda p: None,
        log_artifacts=lambda p: None,
        log_figure=lambda fig, name: None,
        set_tags=lambda tags: None,
        get_tracking_uri=lambda: "http://localhost:5000",
        end_run=lambda status="FINISHED": calls.__setitem__("status", status),
    )

    import sys

    monkeypatch.setitem(sys.modules, "mlflow", dummy)

    tracker = MLflowTracker(experiment_name="exp")
    tracker.log_params({"a": 1, "b": [1, 2]})
    tracker.log_metrics({"loss": 0.1}, step=2)
    tracker.finish("FINISHED")

    assert calls["params"]["a"] == 1
    assert calls["params"]["b"] == "[1, 2]"
    assert calls["metrics"][0]["loss"] == 0.1
    assert calls["metrics"][1] == 2
    assert tracker.get_run_id() == "run-123"
    assert tracker.get_run_url() is not None
    assert calls["status"] == "FINISHED"
