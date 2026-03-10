"""CLI entrypoint for supervised deep regime detector training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.deep import DeepRegimeDetector


def _load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)
    label_col = "label" if "label" in df.columns else df.columns[0]
    return df[label_col].astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep-learning regime detector")
    parser.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--labels", help="Path to labels CSV (timestamp,label)")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", required=True, help="Output detector path (.pkl)")
    args = parser.parse_args()

    ohlcv = pd.read_csv(args.ohlcv, index_col=0, parse_dates=True)

    if args.labels:
        labels = _load_series(Path(args.labels))
    else:
        pseudo = ClusteringRegimeDetector(n_clusters=3, method="kmeans").fit(ohlcv, window=args.window)
        labels = pseudo.predict(ohlcv)

    detector = DeepRegimeDetector(
        model_type=args.model_type,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window=args.window,
    ).fit(ohlcv, labels=labels)

    detector.save(args.output)

    preds = detector.predict(ohlcv).dropna()
    aligned_labels = labels.reindex(preds.index).astype(str)
    accuracy = float((preds.astype(str) == aligned_labels).mean()) if len(preds) else 0.0
    print(f"Validation-like in-sample accuracy: {accuracy:.4f}")
    print(f"Saved deep detector to: {args.output}")


if __name__ == "__main__":
    main()
