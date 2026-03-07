"""Utilities for listing and comparing tracked experiment runs."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mlflow is required for results tooling") from exc
    return mlflow


def list_runs(experiment_name: str | None = None, tags: dict[str, str] | None = None) -> pd.DataFrame:
    mlflow = _mlflow()
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()
        experiments = [experiment.experiment_id]
    else:
        experiments = None

    query = []
    if tags:
        for key, value in tags.items():
            query.append(f"tags.{key} = '{value}'")
    filter_string = " and ".join(query) if query else None
    return mlflow.search_runs(experiment_ids=experiments, filter_string=filter_string)


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    mlflow = _mlflow()
    rows = []
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        row = {"run_id": run_id, **run.data.params, **run.data.metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "run_id" and pd.api.types.is_numeric_dtype(df[c])]
    return {
        "table": df,
        "summary": df[numeric_cols].describe().to_dict() if numeric_cols else {},
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment result utilities")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--compare", nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.compare:
        out = compare_runs(args.compare)
        print(out["table"].to_string(index=False))
        return 0

    df = list_runs(experiment_name=args.experiment_name)
    print(df.to_string(index=False) if not df.empty else "No runs found")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
