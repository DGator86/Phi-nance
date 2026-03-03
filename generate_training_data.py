#!/usr/bin/env python3
"""
generate_training_data.py
--------------------------
Downloads OHLCV data, runs the FULL MFT RegimeEngine pipeline, and saves
the complete feature set as historical_regime_features.csv.

Feature set includes:
  - Raw FeatureEngine features (log_return, rv_30, d_lambda, ...)
  - TaxonomyEngine sticky logits per node (Kingdom / Phylum / Class / Order / Family / Genus)
  - ProbabilityField regime probabilities (8 bins: TREND_UP, TREND_DN, ...)
  - Mixer composite signal + confidence metrics (c_field, c_consensus, c_liquidity)
  - Expert indicator signals + validity weights
  - ProjectionEngine AR(1) expected values
  - Direction label (1 = next close > current, 0 = otherwise)

Usage:
    python generate_training_data.py
    python generate_training_data.py --symbol QQQ --start 2012-01-01 --end 2024-12-31
    python generate_training_data.py --output custom_output.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    cfg_path = os.path.join("regime_engine", "config.yaml")
    if not os.path.exists(cfg_path):
        print(f"ERROR: config not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance", file=sys.stderr)
        sys.exit(1)

    print(f"  Downloading {symbol} ({start} → {end})...")
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print(f"ERROR: No data returned for {symbol}", file=sys.stderr)
        sys.exit(1)

    # Flatten MultiIndex columns (yfinance ≥ 0.2 style)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    print(f"  Downloaded {len(raw):,} bars  ({raw.index[0].date()} → {raw.index[-1].date()})")
    return raw


# ---------------------------------------------------------------------------
# Main feature extraction via full MFT pipeline
# ---------------------------------------------------------------------------
def build_feature_dataframe(ohlcv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run the full RegimeEngine pipeline and return a wide DataFrame
    containing all raw features + regime + blending + confidence columns.
    """
    from regime_engine.scanner import RegimeEngine

    print("  Initializing RegimeEngine...")
    engine = RegimeEngine(cfg)

    print("  Running full MFT pipeline (this may take 30-60 seconds)...")
    out = engine.run(ohlcv)

    # ── 1. Raw features ────────────────────────────────────────────────────
    feat_df = out["features"].copy()
    print(f"    Raw features: {feat_df.shape[1]} columns")

    # ── 2. Taxonomy logits ─────────────────────────────────────────────────
    logits_df = out["logits"].copy()
    logits_df.columns = [f"logit_{c}" for c in logits_df.columns]
    print(f"    Taxonomy logits: {logits_df.shape[1]} nodes")

    # ── 3. Regime probabilities ────────────────────────────────────────────
    regime_df = out["regime_probs"].copy()
    regime_df.columns = [f"prob_{c}" for c in regime_df.columns]
    # One-hot dominant regime
    dominant = out["regime_probs"].idxmax(axis=1)
    for regime in out["regime_probs"].columns:
        regime_df[f"regime_{regime}"] = (dominant == regime).astype(int)
    print(f"    Regime probs: {len(out['regime_probs'].columns)} bins")

    # ── 4. Mixer composite score + confidence ──────────────────────────────
    mix_df = out["mix"].copy()
    mix_cols = [c for c in mix_df.columns if c in (
        "composite_signal", "score", "c_field", "c_consensus", "c_liquidity",
        "blend", "s_linear", "s_interaction",
    )]
    mix_df = mix_df[mix_cols] if mix_cols else mix_df
    print(f"    Mixer outputs: {mix_df.shape[1]} columns")

    # ── 5. Indicator signals + validity weights ────────────────────────────
    signals_df = out["signals"].copy()
    signals_df.columns = [f"sig_{c}" for c in signals_df.columns]
    weights_df = out["weights"].copy()
    weights_df.columns = [f"wt_{c}" for c in weights_df.columns]
    print(f"    Indicators: {len(out['signals'].columns)} signals + weights")

    # ── 6. Projections (AR(1) expected values) ─────────────────────────────
    proj_expected = out["projections"]["expected"].copy()
    proj_expected.columns = [f"proj_{c}" for c in proj_expected.columns]
    proj_variance = out["projections"]["variance"].copy()
    proj_variance.columns = [f"projvar_{c}" for c in proj_variance.columns]
    print(f"    Projections: {proj_expected.shape[1]} expected + variance columns")

    # ── 7. Direction label ────────────────────────────────────────────────
    close = ohlcv["close"].values
    direction = np.zeros(len(close), dtype=int)
    direction[:-1] = (close[1:] > close[:-1]).astype(int)
    direction[-1] = 0  # last bar has no next close → 0

    # ── Align all DataFrames on the same index ────────────────────────────
    base_index = feat_df.index

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(base_index)

    combined = pd.concat(
        [
            feat_df,
            _align(logits_df),
            _align(regime_df),
            _align(mix_df),
            _align(signals_df),
            _align(weights_df),
            _align(proj_expected),
            _align(proj_variance),
        ],
        axis=1,
    )

    combined["direction"] = direction[:len(combined)]

    # Drop the last row (direction=0 since no next bar) and any NaN rows
    combined = combined.iloc[:-1].dropna()

    print(f"  Final dataset: {len(combined):,} rows × {combined.shape[1]} columns")
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate MFT training data")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start",  default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",    default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="historical_regime_features.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phi-nance MFT Training Data Generator")
    print("=" * 60)

    cfg = _load_config()
    ohlcv = _download(args.symbol, args.start, args.end)

    print(f"\nRunning full MFT pipeline on {args.symbol}...")
    df = build_feature_dataframe(ohlcv, cfg)

    df.to_csv(args.output, index=False)
    print(f"\n✅  Saved → {args.output}")
    print(f"     Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    up = df["direction"].sum()
    down = len(df) - up
    print(f"   Labels: {up:,} UP ({up/len(df):.1%})  |  {down:,} DOWN ({down/len(df):.1%})")
    print(f"\nNext step: python train_ml_classifier.py")


if __name__ == "__main__":
    main()
