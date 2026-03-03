"""
Train ML Direction Classifiers
--------------------------------
Trains scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
and LightGBM classifiers on historical regime features, then saves the
fitted models to the `models/` directory.

Prerequisites
-------------
You need a CSV file of historical regime features + direction labels.
The easiest way to generate it is:

    1. Run a backtest using FeatureEngine on your historical OHLCV data.
    2. Label each day:  1 if next-day close > today's close, else 0.
    3. Save as `historical_regime_features.csv`.

Expected CSV columns (minimum):
    All columns from FeatureEngine.FEATURE_COLS plus a `direction` column
    where 1 = UP and 0 = DOWN.

Usage
-----
    # Train all sklearn models + LightGBM (default)
    python train_ml_classifier.py

    # Train only a specific sklearn model
    python train_ml_classifier.py --model random_forest

    # Use a custom data file
    python train_ml_classifier.py --data my_features.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from regime_engine.features import FeatureEngine


MODELS_DIR = "models"
DEFAULT_DATA_FILE = "historical_regime_features.csv"
FEATURE_COLS = FeatureEngine.FEATURE_COLS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")
        print(
            "\nTo generate training data:\n"
            "  1. Fetch historical OHLCV data for your symbol.\n"
            "  2. Run FeatureEngine to compute features.\n"
            "  3. Add a 'direction' column: 1 if next-day return > 0, else 0.\n"
            "  4. Save the result as 'historical_regime_features.csv'.\n"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Validate required columns
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        print(f"WARNING: Missing feature columns (will be zero-filled): {missing_feats}")
        for col in missing_feats:
            df[col] = 0.0

    if "direction" not in df.columns:
        print("ERROR: CSV must have a 'direction' column (1=UP, 0=DOWN).")
        sys.exit(1)

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["direction"].astype(int).values
    print(f"Class balance — UP: {y.sum():,}  DOWN: {(y == 0).sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_sklearn_models(X: pd.DataFrame, y: np.ndarray) -> None:
    """Train RF, GB, and Logistic Regression classifiers and save them."""
    from regime_engine.ml_classifier import DirectionClassifier
    os.makedirs(MODELS_DIR, exist_ok=True)

    n_train = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:n_train], y[:n_train]
    X_val,   y_val   = X.iloc[n_train:], y[n_train:]

    for model_type, filename in [
        ("random_forest",      "classifier_rf.pkl"),
        ("gradient_boosting",  "classifier_gb.pkl"),
        ("logistic",           "classifier_lr.pkl"),
    ]:
        print(f"\n── Training {model_type} ──")
        clf = DirectionClassifier(model_type=model_type)
        clf.train(X_train, y_train)

        # Quick validation accuracy
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        # Re-use the fitted scaler from the clf
        X_val_scaled = clf.scaler.transform(X_val)
        y_pred = clf.model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        print(f"  Val accuracy: {acc:.3f}  ({y_pred.sum()} UP / {(y_pred==0).sum()} DOWN predicted)")

        out_path = os.path.join(MODELS_DIR, filename)
        clf.save(out_path)


def train_lightgbm_model(X: pd.DataFrame, y: np.ndarray) -> None:
    """Train LightGBM classifier and save it."""
    from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n── Training LightGBM ──")
    clf = LightGBMDirectionClassifier(num_leaves=31, learning_rate=0.05, num_rounds=500)
    clf.train(X, y, validation_split=0.2)

    # Feature importance
    try:
        imp = clf.get_feature_importance(top_n=10)
        print("\nTop-10 features by gain:")
        print(imp.to_string(index=False))
    except Exception:
        pass

    out_path = os.path.join(MODELS_DIR, "classifier_lgb.txt")
    clf.save(out_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phi-nance ML classifiers")
    parser.add_argument(
        "--model",
        choices=["all", "random_forest", "gradient_boosting", "logistic", "lightgbm"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_FILE,
        help=f"Path to feature CSV (default: {DEFAULT_DATA_FILE})",
    )
    args = parser.parse_args()

    X, y = load_data(args.data)

    if args.model in ("all", "random_forest", "gradient_boosting", "logistic"):
        train_sklearn_models(X, y)

    if args.model in ("all", "lightgbm"):
        train_lightgbm_model(X, y)

    print(f"\nDone! Model files saved to '{MODELS_DIR}/'")


if __name__ == "__main__":
    main()
