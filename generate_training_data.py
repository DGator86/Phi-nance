"""
Generate Training Data
-----------------------
Downloads historical SPY OHLCV data via yfinance, computes regime features
using FeatureEngine, labels each day with next-day direction, and saves
the result to `historical_regime_features.csv`.

Run once before `train_ml_classifier.py`:
    python generate_training_data.py

Optional flags:
    --symbol  QQQ          (default: SPY)
    --start   2010-01-01   (default: 2015-01-01)
    --end     2024-12-31   (default: 2024-12-31)
    --out     my_data.csv  (default: historical_regime_features.csv)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import yaml
import os

# ---------------------------------------------------------------------------
# Check yfinance is available
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run:  pip install yfinance")
    sys.exit(1)

from regime_engine.features import FeatureEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate historical_regime_features.csv for ML training"
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--start",  default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",    default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--out",
        default="historical_regime_features.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # ── 1. Load config ──────────────────────────────────────────────────
    cfg_path = os.path.join("regime_engine", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        features_cfg = cfg.get("features", {})
    else:
        print(f"Warning: {cfg_path} not found — using default feature config.")
        features_cfg = {}   # FeatureEngine will use defaults

    # ── 2. Download OHLCV ───────────────────────────────────────────────
    print(f"Downloading {args.symbol} daily OHLCV {args.start} → {args.end}...")
    raw = yf.download(args.symbol, start=args.start, end=args.end, interval="1d",
                      auto_adjust=True, progress=False)

    if raw.empty:
        print("ERROR: No data returned from yfinance. Check symbol / dates.")
        sys.exit(1)

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    # Rename 'adj close' → 'close' if needed
    if "adj close" in raw.columns and "close" not in raw.columns:
        raw = raw.rename(columns={"adj close": "close"})

    print(f"Downloaded {len(raw):,} bars.")

    # ── 3. Compute features ─────────────────────────────────────────────
    print("Computing regime features...")
    engine  = FeatureEngine(features_cfg)
    feats   = engine.compute(raw)

    # ── 4. Add close price & direction label ────────────────────────────
    feats["close"] = raw["close"].values

    # direction = 1 if NEXT day's close > today's close, else 0
    feats["direction"] = (raw["close"].shift(-1) > raw["close"]).astype(int).values

    # ── 5. Drop warmup / lookahead ──────────────────────────────────────
    feats = feats.dropna()                # removes warmup NaNs
    feats = feats.iloc[:-1]              # drop last row (no next-day label)

    up_count   = int(feats["direction"].sum())
    down_count = len(feats) - up_count
    print(
        f"Rows after cleaning: {len(feats):,}  |  UP: {up_count:,}  DOWN: {down_count:,}"
    )

    # ── 6. Save ─────────────────────────────────────────────────────────
    feats.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")
    print("Next step:")
    print("    python train_ml_classifier.py")


if __name__ == "__main__":
    main()
