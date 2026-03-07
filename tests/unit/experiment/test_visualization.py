from __future__ import annotations

from dataclasses import dataclass

import matplotlib.figure
import plotly.graph_objects as go

from phinance.experiment import visualization


@dataclass
class DummyMetricPoint:
    step: int
    value: float
    timestamp: int = 0


@dataclass
class DummyRunInfo:
    run_id: str


@dataclass
class DummyRunData:
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]


@dataclass
class DummyRun:
    info: DummyRunInfo
    data: DummyRunData


class DummyClient:
    def __init__(self) -> None:
        self._runs = {
            "run-1": DummyRun(
                info=DummyRunInfo(run_id="run-1"),
                data=DummyRunData(
                    metrics={"reward": 1.5, "val_reward": 1.1},
                    params={"lr": "0.01", "depth": "4"},
                    tags={"sweep_id": "sweep-1"},
                ),
            ),
            "run-2": DummyRun(
                info=DummyRunInfo(run_id="run-2"),
                data=DummyRunData(
                    metrics={"reward": 1.8, "val_reward": 1.4},
                    params={"lr": "0.02", "depth": "6"},
                    tags={"sweep_id": "sweep-1"},
                ),
            ),
        }

    def get_run(self, run_id: str) -> DummyRun:
        return self._runs[run_id]

    def get_metric_history(self, run_id: str, metric_name: str) -> list[DummyMetricPoint]:
        base = self._runs[run_id].data.metrics.get(metric_name, 0.0)
        return [DummyMetricPoint(step=0, value=base - 0.5), DummyMetricPoint(step=1, value=base)]

    def search_runs(self, experiment_ids=None, filter_string: str | None = None, max_results: int = 5000):
        if filter_string and "sweep-1" in filter_string:
            return [self._runs["run-1"], self._runs["run-2"]]
        return []



def _patch_client(monkeypatch):
    monkeypatch.setattr(visualization, "_get_mlflow_client", lambda: DummyClient())



def test_get_run_data(monkeypatch) -> None:
    _patch_client(monkeypatch)
    data = visualization.get_run_data("run-1")
    assert data["run_id"] == "run-1"
    assert "reward" in data["metrics"]
    assert "reward" in data["metric_history"]



def test_plot_learning_curve(monkeypatch) -> None:
    _patch_client(monkeypatch)
    fig = visualization.plot_learning_curve(["run-1", "run-2"], metric="reward")
    assert isinstance(fig, matplotlib.figure.Figure)



def test_plot_trial_comparison(monkeypatch) -> None:
    _patch_client(monkeypatch)
    fig = visualization.plot_trial_comparison("sweep-1", metric="val_reward")
    assert isinstance(fig, matplotlib.figure.Figure)



def test_parallel_coordinates(monkeypatch) -> None:
    _patch_client(monkeypatch)
    fig = visualization.parallel_coordinates("sweep-1", metric="val_reward")
    assert isinstance(fig, go.Figure)



def test_parameter_importance(monkeypatch) -> None:
    _patch_client(monkeypatch)
    fig = visualization.parameter_importance("sweep-1", metric="val_reward")
    assert isinstance(fig, matplotlib.figure.Figure)
