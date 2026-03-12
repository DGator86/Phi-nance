#!/usr/bin/env python3
"""Analyze backtest results DB and export best model per dominant regime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from phi.learning.results_db import ResultsDB


def dominant_regime(sequence: list[str]) -> str | None:
    if not sequence:
        return None
    return Counter(sequence).most_common(1)[0][0]


def _to_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def analyze_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data = df.copy()
    data["dominant_regime"] = data["regime_sequence"].apply(dominant_regime)
    data["parameters"] = data["parameters"].apply(_to_params)

    perf = (
        data.groupby(["dominant_regime", "strategy_name"], dropna=False)["sharpe_ratio"]
        .mean()
        .unstack(fill_value=0.0)
        .sort_index()
    )

    idx = data.groupby("dominant_regime")["sharpe_ratio"].idxmax()
    best_per_regime = data.loc[idx].sort_values("dominant_regime")

    longcall = data[data["strategy_name"] == "LongCallStrategy"].copy()
    longcall["delta_threshold"] = longcall["parameters"].apply(lambda x: x.get("delta_threshold"))
    longcall["min_volume"] = longcall["parameters"].apply(lambda x: x.get("min_volume"))

    return perf, best_per_regime, longcall


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze options backtest results")
    parser.add_argument("--db-path", default="backtest_results.db")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="configs/best_models_per_regime.yaml")
    args = parser.parse_args()

    db = ResultsDB(db_path=args.db_path)
    runs = db.get_runs(limit=args.limit)

    perf, best_per_regime, longcall = analyze_runs(runs)
    print("Columns:", list(runs.columns))
    print("\nPerformance heatmap table (mean Sharpe):")
    print(perf)

    if not longcall.empty:
        corr = longcall[["delta_threshold", "min_volume", "sharpe_ratio"]].corr(numeric_only=True)
        print("\nLongCall parameter correlation with Sharpe:")
        print(corr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict[str, Any]] = {}
    for _, row in best_per_regime.iterrows():
        regime = row.get("dominant_regime")
        if not isinstance(regime, str) or not regime:
            continue
        mapping[regime] = {
            "strategy_name": row["strategy_name"],
            "parameters": row["parameters"],
            "sharpe_ratio": float(row["sharpe_ratio"]),
            "symbol": row.get("symbol"),
            "created_at": row.get("created_at"),
        }

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(mapping, handle, sort_keys=True)

    print(f"\nSaved best models by regime to: {output_path}")


if __name__ == "__main__":
    main()
