"""MLflow experiment tracker implementation."""

from __future__ import annotations

from typing import Any, Optional

from phinance.experiment.tracker import ExperimentTracker, ensure_artifact_exists


class MLflowTracker(ExperimentTracker):
    """MLflow-backed experiment tracker."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
        parent_run_id: str | None = None,
        **_: Any,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("mlflow is not installed. Install with `pip install mlflow`.") from exc

        self._mlflow = mlflow
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)
        if parent_run_id:
            self._run = self._mlflow.start_run(run_name=run_name, nested=nested, parent_run_id=parent_run_id)
        else:
            self._run = self._mlflow.start_run(run_name=run_name, nested=nested)
        if tags:
            self.set_tags(tags)

    def log_params(self, params: dict[str, Any]) -> None:
        serialised = {k: str(v) if isinstance(v, (list, dict, tuple, set)) else v for k, v in params.items()}
        self._mlflow.log_params(serialised)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items()}
        self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, local_path: str) -> None:
        artifact_path = ensure_artifact_exists(local_path)
        if artifact_path.is_dir():
            self._mlflow.log_artifacts(str(artifact_path))
        else:
            self._mlflow.log_artifact(str(artifact_path))

    def log_figure(self, figure: Any, name: str) -> None:
        self._mlflow.log_figure(figure, name)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags(tags)

    def get_run_id(self) -> str:
        return self._run.info.run_id

    def get_run_url(self) -> Optional[str]:
        uri = self._mlflow.get_tracking_uri()
        if uri.startswith("file:"):
            return None
        return f"{uri.rstrip('/')}/#/experiments/{self._run.info.experiment_id}/runs/{self.get_run_id()}"

    def finish(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)
