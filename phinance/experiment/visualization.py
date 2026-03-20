"""Visualization utilities for analysing MLflow tracked experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def _get_mlflow_client() -> Any:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mlflow is required for visualization tooling") from exc
    return MlflowClient()


def _coerce_numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_run_data(run_id: str) -> dict[str, Any]:
    """Fetch run metadata and metric history for a run id."""
    client = _get_mlflow_client()
    run = client.get_run(run_id)

    metric_history: dict[str, list[dict[str, float | int | None]]] = {}
    for metric_name in run.data.metrics:
        history = client.get_metric_history(run_id, metric_name)
        metric_history[metric_name] = [
            {"step": point.step, "value": point.value, "timestamp": point.timestamp}
            for point in history
        ]

    return {
        "run_id": run_id,
        "metrics": dict(run.data.metrics),
        "metric_history": metric_history,
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def plot_learning_curve(
    run_ids: list[str],
    metric: str = "reward",
    title: str | None = None,
    interactive: bool = False,
):
    """Plot one or many run metric histories over steps."""
    client = _get_mlflow_client()
    rows: list[dict[str, Any]] = []

    for run_id in run_ids:
        history = client.get_metric_history(run_id, metric)
        for point in history:
            rows.append({"run_id": run_id, "step": point.step, "value": point.value})

    if not rows:
        raise ValueError(f"No metric history found for metric '{metric}'.")

    df = pd.DataFrame(rows).sort_values(["run_id", "step"])

    if interactive:
        fig = px.line(df, x="step", y="value", color="run_id", title=title or f"{metric} learning curve")
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_id, group in df.groupby("run_id"):
        ax.plot(group["step"], group["value"], label=run_id)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} learning curve")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _get_child_runs(client: Any, sweep_id: str) -> list[Any]:
    filters = [f"tags.mlflow.parentRunId = '{sweep_id}'", f"tags.sweep_id = '{sweep_id}'"]
    for filter_string in filters:
        runs = client.search_runs(experiment_ids=None, filter_string=filter_string, max_results=5000)
        if runs:
            return list(runs)
    return []


def _extract_final_metric(runs: list[Any], metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metric_value = run.data.metrics.get(metric)
        if metric_value is None:
            continue
        row = {
            "run_id": run.info.run_id,
            "metric": metric_value,
        }
        for key, value in run.data.params.items():
            row[f"param.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_trial_comparison(
    sweep_id: str,
    metric: str = "val_reward",
    top_k: int | None = None,
    interactive: bool = False,
):
    """Compare child runs from a sweep by final metric."""
    client = _get_mlflow_client()
    runs = _get_child_runs(client, sweep_id)
    if not runs:
        raise ValueError(f"No child runs found for sweep '{sweep_id}'.")

    df = _extract_final_metric(runs, metric)
    if df.empty:
        raise ValueError(f"No '{metric}' metric found for sweep '{sweep_id}'.")

    df = df.sort_values("metric", ascending=False)
    if top_k is not None:
        df = df.head(top_k)

    if interactive:
        return px.bar(df, x="run_id", y="metric", title=f"Sweep {sweep_id}: {metric} by run")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["run_id"], df["metric"])
    ax.set_xlabel("Run ID")
    ax.set_ylabel(metric)
    ax.set_title(f"Sweep {sweep_id}: {metric} by run")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def parallel_coordinates(
    sweep_id: str,
    params: list[str] | None = None,
    metric: str = "val_reward",
):
    """Build an interactive parameter-performance parallel coordinate plot."""
    client = _get_mlflow_client()
    runs = _get_child_runs(client, sweep_id)
    if not runs:
        raise ValueError(f"No child runs found for sweep '{sweep_id}'.")

    df = _extract_final_metric(runs, metric)
    if df.empty:
        raise ValueError(f"No '{metric}' metric found for sweep '{sweep_id}'.")

    param_columns = [col for col in df.columns if col.startswith("param.")]
    if params is not None:
        wanted = {f"param.{name}" for name in params}
        param_columns = [col for col in param_columns if col in wanted]

    if not param_columns:
        raise ValueError("No parameter columns found for parallel coordinates plot.")

    working = df[param_columns + ["metric"]].copy()
    for col in param_columns:
        numeric_values = working[col].map(_coerce_numeric)
        if numeric_values.notna().all():
            working[col] = numeric_values
        else:
            working[col] = working[col].astype("category").cat.codes

    return px.parallel_coordinates(working, color="metric", dimensions=param_columns, title=f"Sweep {sweep_id}")


def parameter_importance(sweep_id: str, metric: str = "val_reward"):
    """Estimate parameter importance via absolute Pearson correlation."""
    client = _get_mlflow_client()
    runs = _get_child_runs(client, sweep_id)
    if not runs:
        raise ValueError(f"No child runs found for sweep '{sweep_id}'.")

    df = _extract_final_metric(runs, metric)
    if df.empty:
        raise ValueError(f"No '{metric}' metric found for sweep '{sweep_id}'.")

    param_columns = [col for col in df.columns if col.startswith("param.")]
    if not param_columns:
        raise ValueError("No parameter columns found for importance plot.")

    working = df[param_columns + ["metric"]].copy()
    for col in param_columns:
        numeric_values = working[col].map(_coerce_numeric)
        if numeric_values.notna().all():
            working[col] = numeric_values
        else:
            working[col] = working[col].astype("category").cat.codes

    corr = working.corr(numeric_only=True)["metric"].drop(labels=["metric"], errors="ignore").abs()
    corr = corr.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(corr.index, corr.values)
    ax.set_xlabel("Parameter")
    ax.set_ylabel(f"|corr| with {metric}")
    ax.set_title(f"Parameter importance for sweep {sweep_id}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
