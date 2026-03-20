"""Experiment tracker abstractions and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking backends."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for a run."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for a run."""

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log artifact file or directory."""

    @abstractmethod
    def log_figure(self, figure: Any, name: str) -> None:
        """Log a figure artifact."""

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags for a run."""

    @abstractmethod
    def get_run_id(self) -> str:
        """Return unique run id."""

    @abstractmethod
    def get_run_url(self) -> Optional[str]:
        """Return run URL when available."""

    @abstractmethod
    def finish(self, status: str = "FINISHED") -> None:
        """Finalize run resources."""


class NoOpTracker(ExperimentTracker):
    """Tracker implementation that performs no logging."""

    def __init__(self) -> None:
        self._run_id = "noop"

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        return None

    def log_artifact(self, local_path: str) -> None:
        return None

    def log_figure(self, figure: Any, name: str) -> None:
        return None

    def set_tags(self, tags: dict[str, str]) -> None:
        return None

    def get_run_id(self) -> str:
        return self._run_id

    def get_run_url(self) -> Optional[str]:
        return None

    def finish(self, status: str = "FINISHED") -> None:
        return None


class WandBTrackerUnavailable(NoOpTracker):
    """Placeholder for future Weights & Biases support."""

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self._run_id = "wandb-unavailable"


def create_tracker(backend: str, **kwargs: Any) -> ExperimentTracker:
    """Create a tracker backend by name."""
    normalized = backend.lower().strip()
    if normalized in {"none", "noop", "disabled"}:
        return NoOpTracker()
    if normalized == "mlflow":
        from phinance.experiment.mlflow_tracker import MLflowTracker

        return MLflowTracker(**kwargs)
    if normalized in {"wandb", "weights_biases", "weights-and-biases"}:
        return WandBTrackerUnavailable(**kwargs)
    raise ValueError(f"Unsupported experiment tracker backend: {backend}")


def ensure_artifact_exists(path: str) -> Path:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return artifact_path
