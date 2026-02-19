"""
Feature Importance Analysis
----------------------------
Loads the trained LightGBM model and prints / plots the feature importance
table.  Helps you understand which regime features drive the predictions
most and guides further feature engineering.

Usage:
    python analyze_features.py
    python analyze_features.py --model models/classifier_lgb.txt --top 20
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LightGBM feature importance")
    parser.add_argument(
        "--model",
        default="models/classifier_lgb.txt",
        help="Path to LightGBM model file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top features to display",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(
            f"ERROR: Model file not found at '{args.model}'.\n"
            "Train first with: python train_ml_classifier.py --model lightgbm"
        )
        return

    from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier

    clf = LightGBMDirectionClassifier()
    clf.load(args.model)

    print(f"\nTop-{args.top} feature importances (by information gain):\n")
    imp_df = clf.get_feature_importance(top_n=args.top)
    print(imp_df.to_string(index=False))

    print(
        "\nTip: Features with high gain are the ones the model splits on most "
        "informatively.\nConsider engineering more variants of the top features."
    )


if __name__ == "__main__":
    main()
