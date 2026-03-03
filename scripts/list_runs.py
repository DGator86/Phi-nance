#!/usr/bin/env python3
"""
scripts/list_runs.py
=====================
CLI utility to list and inspect saved backtest runs.

Usage
-----
    python scripts/list_runs.py               # list 20 most recent
    python scripts/list_runs.py --limit 50    # list up to 50
    python scripts/list_runs.py --run-id <id> # show full detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List / inspect saved backtest runs")
    p.add_argument("--limit",  type=int, default=20, help="Max runs to list")
    p.add_argument("--run-id", default="",            help="Show full detail for a run ID")
    p.add_argument("--json",   action="store_true",   help="Output raw JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from phinance.storage import RunHistory
    from phinance.exceptions import RunNotFoundError

    history = RunHistory()

    if args.run_id:
        try:
            run = history.load_run(args.run_id)
        except RunNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

        if args.json:
            print(json.dumps({"run_id": run.run_id, "config": run.config, "results": run.results},
                             indent=2, default=str))
            return 0

        print(f"\n{'='*60}")
        print(f"  Run:  {run.run_id}")
        print(f"{'='*60}")
        print(f"  Symbols   : {', '.join(run.symbols)}")
        cfg = run.config
        print(f"  Period    : {cfg.get('start_date', '?')} → {cfg.get('end_date', '?')}")
        print(f"  Timeframe : {cfg.get('timeframe', '?')}")
        print(f"  Blend     : {cfg.get('blend_method', '?')}")
        print(f"  PhiAI     : {cfg.get('phiai_enabled', False)}")
        print()
        r = run.results
        print(f"  Total Return  : {float(r.get('total_return', 0)):+.2%}")
        print(f"  CAGR          : {float(r.get('cagr', 0)):+.2%}")
        print(f"  Max Drawdown  : -{float(r.get('max_drawdown', 0)):.2%}")
        print(f"  Sharpe        : {float(r.get('sharpe', 0)):.3f}")
        print(f"  Win Rate      : {float(r.get('win_rate', 0)):.1%}")
        print(f"  Total Trades  : {r.get('total_trades', 0)}")
        if run.trades is not None:
            print(f"  Trades on disk: {len(run.trades)}")
        print()
        return 0

    # ── List mode ─────────────────────────────────────────────────────────
    runs = history.list_runs(limit=args.limit)

    if not runs:
        print("No saved runs found.")
        return 0

    if args.json:
        print(json.dumps(runs, indent=2, default=str))
        return 0

    hdr = (f"{'Run ID':<26} {'Symbols':<10} {'TF':<5} "
           f"{'Return':>8} {'Sharpe':>7} {'DD':>7} {'Trades':>7}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for r in runs:
        cfg  = r.get("config", {})
        res  = r.get("results", {})
        syms = ",".join(cfg.get("symbols", []))[:9]
        tf   = cfg.get("timeframe", "")[:4]
        ret  = f"{float(res.get('total_return', 0)):+.1%}"
        sharpe = f"{float(res.get('sharpe', 0)):.2f}"
        dd   = f"-{float(res.get('max_drawdown', 0)):.1%}"
        trades = str(res.get("total_trades", ""))
        print(f"{r['run_id']:<26} {syms:<10} {tf:<5} "
              f"{ret:>8} {sharpe:>7} {dd:>7} {trades:>7}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
