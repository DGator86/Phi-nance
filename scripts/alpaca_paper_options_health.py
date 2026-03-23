#!/usr/bin/env python3
"""
Alpaca **paper** account health check for options workflows.

- Reads API keys **only** from the environment (never hard-code secrets).
- Prints account snapshot (equity, cash, buying power, PDT flag).
- Optional: **--confirm-mtf** runs Phi-nance MTF regime matrix on daily OHLCV
  (Unusual Whales → yfinance), same logic as the options signal card.

Usage
-----
  set ALPACA_API_KEY=...
  set ALPACA_SECRET_KEY=...
  set ALPACA_BASE_URL=https://paper-api.alpaca.markets

  python scripts/alpaca_paper_options_health.py
  python scripts/alpaca_paper_options_health.py --confirm-mtf --symbol SPY

Also accepts BROKER_API_KEY / BROKER_SECRET_KEY / BROKER_BASE_URL (phi.config style).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _keys() -> tuple[str, str, str]:
    key = (os.getenv("ALPACA_API_KEY") or os.getenv("BROKER_API_KEY") or "").strip()
    secret = (os.getenv("ALPACA_SECRET_KEY") or os.getenv("BROKER_SECRET_KEY") or "").strip()
    base = (
        os.getenv("ALPACA_BASE_URL")
        or os.getenv("BROKER_BASE_URL")
        or "https://paper-api.alpaca.markets"
    ).strip()
    return key, secret, base.rstrip("/")


def _alpaca_connect():
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise SystemExit("Install alpaca-py: pip install alpaca-py") from exc
    key, secret, base = _keys()
    if not key or not secret:
        raise SystemExit(
            "Missing ALPACA_API_KEY + ALPACA_SECRET_KEY (or BROKER_* aliases). See docs/alpaca_paper_options.md"
        )
    paper = "paper-api" in base
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def run_account_snapshot() -> int:
    client = _alpaca_connect()
    acct = client.get_account()
    print("=== Alpaca paper account ===")
    print(f"  id:                 {acct.id}")
    print(f"  status:             {acct.status}")
    print(f"  equity:             {float(acct.equity):,.2f}")
    print(f"  cash:               {float(acct.cash):,.2f}")
    print(f"  buying_power:       {float(acct.buying_power):,.2f}")
    print(f"  portfolio_value:    {float(acct.portfolio_value):,.2f}")
    print(f"  pattern_day_trader: {acct.pattern_day_trader}")
    print(f"  trading_blocked:    {acct.trading_blocked}")
    if hasattr(acct, "daytrade_count"):
        print(f"  daytrade_count:     {acct.daytrade_count}")
    return 0


def run_confirm_mtf(symbol: str, lookback_days: int) -> int:
    from datetime import date, timedelta

    import pandas as pd

    from phi.data import fetch_ohlcv_uw_then_yf
    from phi.regime.mtf_matrix import build_regime_matrix, confluence_score

    sym = symbol.strip().upper()
    end = date.today()
    start = end - timedelta(days=max(lookback_days, 120))
    print(f"\n=== MTF confirmation ({sym}, daily OHLCV via UW → yfinance) ===")
    df, vendor = fetch_ohlcv_uw_then_yf(sym, start.isoformat(), end.isoformat(), timeframe="1D")
    print(f"  bars: {len(df)}  vendor: {vendor}")
    mat, meta = build_regime_matrix(df, min_resampled_bars=25)
    if mat.shape[1] == 0:
        print("  No MTF columns (check history length / resample).")
        print("  skipped:", meta.get("skipped", {}))
        return 1
    last = mat.iloc[-1]
    conf = float(confluence_score(mat).iloc[-1])
    print(f"  timeframes: {list(mat.columns)}")
    print(f"  last bar regimes: {last.to_dict()}")
    print(f"  confluence [-1..+1]: {conf:+.4f}")
    print(
        "\n  Note: With **daily** bars, sub-daily rules (1m–4H) are skipped; "
        "you get 1D / 1W / 1ME-style columns when history allows."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Alpaca paper health + optional MTF check")
    p.add_argument("--confirm-mtf", action="store_true", help="Run MTF matrix on daily OHLCV")
    p.add_argument("--symbol", default="SPY", help="Underlying for MTF check")
    p.add_argument("--lookback-days", type=int, default=420, help="History for MTF OHLCV")
    args = p.parse_args()

    rc = run_account_snapshot()
    if args.confirm_mtf:
        rc = max(rc, run_confirm_mtf(args.symbol, args.lookback_days))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
