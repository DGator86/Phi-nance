import os
import sys
import pandas as pd
import yaml

def check_step(label, fn):
    print(f"[*] Checking: {label}...")
    try:
        fn()
        print(f"[+] SUCCESS: {label}\n")
        return True
    except Exception as e:
        print(f"[-] FAILED: {label}")
        print(f"    Error: {e}\n")
        return False

def test_yfinance():
    import yfinance as yf
    print("    Downloading 5 days of SPY...")
    df = yf.download("SPY", period="5d", progress=False)
    if df.empty:
        raise Exception("Yahoo Finance returned an empty DataFrame. Your VPS IP might be blocked.")
    print(f"    Got {len(df)} rows of data.")

def test_models():
    models = [
        "models/classifier_rf.pkl",
        "models/classifier_gb.pkl",
        "models/classifier_lr.pkl",
        "models/classifier_lgb.txt"
    ]
    missing = []
    for m in models:
        path = os.path.join(os.getcwd(), m)
        if not os.path.exists(path):
            missing.append(m)
    if missing:
        raise Exception(f"Missing model files: {', '.join(missing)}. You need to run training first.")

def test_config():
    path = os.path.join("regime_engine", "config.yaml")
    if not os.path.exists(path):
        raise Exception("regime_engine/config.yaml not found.")
    with open(path) as f:
        yaml.safe_load(f)

def test_imports():
    from regime_engine.scanner import RegimeEngine
    from lumibot.backtesting import YahooDataBacktesting
    from strategies.blended_mft_strategy import BlendedMFTStrategy

def main():
    print("="*60)
    print("  Phi-nance Backtest Diagnostic Tool")
    print("="*60 + "\n")

    steps = [
        ("Imports & Dependencies", test_imports),
        ("regime_engine/config.yaml", test_config),
        ("Yahoo Finance Connectivity (yfinance)", test_yfinance),
        ("ML Model Files", test_models),
    ]

    all_ok = True
    for label, fn in steps:
        if not check_step(label, fn):
            all_ok = False

    print("="*60)
    if all_ok:
        print("  DIAGNOSTICS PASSED: Your environment looks correct.")
    else:
        print("  DIAGNOSTICS FAILED: Please address the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()
