"""
Train RL Agent (PPO)
----------------------
Trains a Stable-Baselines3 PPO agent on TradingEnv and saves the policy.

Prerequisites:
    pip install stable-baselines3 gymnasium

Data:
    Requires a CSV with OHLCV columns AND pre-computed regime feature columns
    (all columns from FeatureEngine.FEATURE_COLS + 'close').
    Generate with:
        df_feats = FeatureEngine(cfg).compute(ohlcv_df)
        df_feats['close'] = ohlcv_df['close']
        df_feats.dropna().to_csv('historical_regime_features.csv')

Usage:
    python train_rl_agent.py
    python train_rl_agent.py --data historical_regime_features.csv --timesteps 200000
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


MODELS_DIR = "models"
DEFAULT_DATA = "historical_regime_features.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phi-nance RL agent (PPO)")
    parser.add_argument("--data",       default=DEFAULT_DATA)
    parser.add_argument("--timesteps",  type=int, default=100_000)
    parser.add_argument("--out",        default=os.path.join(MODELS_DIR, "rl_agent_ppo"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print(
            "stable-baselines3 is required.\n"
            "Install with: pip install stable-baselines3 gymnasium"
        )
        sys.exit(1)

    from regime_engine.rl_trading_env import TradingEnv

    if not os.path.exists(args.data):
        print(
            f"ERROR: Data file not found: {args.data}\n"
            "Generate with FeatureEngine and save a CSV with OHLCV + feature columns."
        )
        sys.exit(1)

    df = pd.read_csv(args.data).dropna()
    print(f"Loaded {len(df):,} rows for RL training.")

    # ── Train / test split (80/20) ──────────────────────────────────────
    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test  = df.iloc[n_train:].reset_index(drop=True)

    train_env = TradingEnv(df_train)
    check_env(train_env, warn=True)

    # ── Train PPO ──────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )

    print(f"\nTraining PPO for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(args.out)
    print(f"Agent saved to {args.out}.zip")

    # ── Evaluate on test set ───────────────────────────────────────────
    print("\nEvaluating on held-out test data...")
    test_env = TradingEnv(df_test)
    obs, _   = test_env.reset()

    for _ in range(len(df_test) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        if terminated or truncated:
            break

    final_value  = info.get("portfolio_value", test_env.initial_cash)
    pct_return   = (final_value / test_env.initial_cash - 1) * 100
    print(f"Test portfolio value:  ${final_value:,.2f}")
    print(f"Test return:            {pct_return:+.1f}%")
    print(f"\nDone! Model at: {args.out}.zip")


if __name__ == "__main__":
    main()
