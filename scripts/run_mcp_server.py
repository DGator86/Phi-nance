#!/usr/bin/env python3
"""
Run Phi-nance as an MCP server so Agent World Model (or any MCP client) can call projection/backtest tools.

Requires: pip install phi-nance[mcp]

  python -m scripts.run_mcp_server --port 8001

Then connect at http://localhost:8001/mcp (streamable HTTP). Use with AWM:
  awm agent --task "Get projection for SPY" --mcp_url http://localhost:8001/mcp ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("MCP SDK required. Install with: pip install phi-nance[mcp]")
        return 1

    p = argparse.ArgumentParser(description="Phi-nance MCP server for AWM / MCP clients")
    p.add_argument("--port", type=int, default=8001, help="Port for streamable HTTP")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bars", help="Bar store root")
    args = p.parse_args()

    # Load .env so providers work if tools need live data
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    from phinence.assignment.engine import AssignmentEngine
    from phinence.composer.composer import Composer
    from phinence.engines.hedge import HedgeEngine
    from phinence.engines.liquidity import LiquidityEngine
    from phinence.engines.regime import RegimeEngine
    from phinence.engines.sentiment import SentimentEngine
    from phinence.store.memory_store import InMemoryBarStore
    from phinence.store.parquet_store import ParquetBarStore
    from phinence.validation.backtest_runner import make_synthetic_bars
    from phinence.mfm.merger import build_mfm
    from phinence.contracts.projection_packet import Horizon
    import pandas as pd
    from datetime import datetime, timezone

    if args.data_root.exists():
        bar_store = ParquetBarStore(args.data_root)
    else:
        bar_store = InMemoryBarStore()
        for t in ["SPY", "QQQ"]:
            bar_store.put_1m_bars(t, make_synthetic_bars(t, "2024-01-01", "2024-06-30", seed=hash(t) % 10000))
    assigner = AssignmentEngine(bar_store)
    composer = Composer()
    engines = {
        "liquidity": LiquidityEngine(),
        "regime": RegimeEngine(),
        "sentiment": SentimentEngine(),
        "hedge": HedgeEngine(),
    }

    mcp = FastMCP(
        "Phi-nance",
        description="Projection and backtest tools: get_projection, list_tickers, run_backtest",
    )

    @mcp.tool()
    def list_tickers() -> list[str]:
        """List ticker symbols available in the bar store."""
        return bar_store.list_tickers() if hasattr(bar_store, "list_tickers") else []

    @mcp.tool()
    def get_projection(ticker: str, as_of_date: str = "") -> str:
        """Get a ProjectionPacket for the given ticker and date (YYYY-MM-DD). Default as_of is today."""
        from phinence.validation.paper_trading import paper_run_daily
        as_of = datetime.now(timezone.utc)
        if as_of_date:
            try:
                as_of = datetime.strptime(as_of_date, "%Y-%m-%d").replace(hour=16, minute=0, second=0, microsecond=0)
            except ValueError:
                return f"Invalid as_of_date: {as_of_date}. Use YYYY-MM-DD."
        packets = paper_run_daily([ticker], bar_store, assigner, engines, composer, as_of=as_of)
        if not packets:
            return f"No projection for {ticker}."
        p = packets[0]
        hp = p.get_horizon(Horizon.DAILY)
        if not hp:
            return p.model_dump_json()
        return (
            f"Ticker={p.ticker} as_of={p.as_of} | "
            f"direction: up={hp.direction.up:.2f} down={hp.direction.down:.2f} flat={hp.direction.flat:.2f} | "
            f"drift_bps={hp.drift_bps:.1f} cone_75_bps={hp.cone.p75_bps:.1f}"
        )

    @mcp.tool()
    def run_backtest(tickers: str = "SPY", start: str = "2024-01-01", end: str = "2024-06-30") -> str:
        """Run walk-forward backtest for given tickers and date range. Returns summary (AUC, cone, gate)."""
        from phinence.validation.walk_forward import WFMode, WalkForwardHarness, expanding_windows
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            ticker_list = ["SPY"]
        mode = WFMode.DAILY
        harness = WalkForwardHarness(mode=mode)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        folds = list(expanding_windows(start_ts, end_ts, mode))
        if not folds:
            return "Not enough date range for a single fold."
        all_metrics = []
        for t in ticker_list:
            for fold in folds:
                m = harness.run_fold(fold, t, bar_store, assigner, engines, composer)
                all_metrics.append(m)
        mean_auc = sum(x["oos_auc"] for x in all_metrics) / len(all_metrics)
        mean_c75 = sum(x["cone_75"] for x in all_metrics) / len(all_metrics)
        gate = "PASS" if harness.gate_passed(all_metrics) else "FAIL"
        return f"Mean OOS AUC: {mean_auc:.3f} | Mean 75% cone: {mean_c75:.2%} | Gate (>=0.52): {gate}"

    print(f"Phi-nance MCP server at http://{args.host}:{args.port}/mcp")
    mcp.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
