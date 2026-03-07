#!/usr/bin/env python3
"""
scripts/run_backtest.py
========================
CLI entry point for running a Phi-nance backtest from the command line.

Usage
-----
    python scripts/run_backtest.py \\
        --symbol SPY \\
        --start  2022-01-01 \\
        --end    2024-12-31 \\
        --tf     1D \\
        --vendor yfinance \\
        --indicators RSI MACD Bollinger \\
        --blend  weighted_sum \\
        --capital 100000 \\
        --phiai

Output
------
Prints a summary table to stdout and saves the run to the local runs/ store.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phi-nance command-line backtest runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol",     default="SPY",  help="Ticker symbol")
    p.add_argument("--start",      default="2022-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",        default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--tf",         default="1D",   help="Timeframe (1D, 1H, 15m …)")
    p.add_argument("--vendor",     default="yfinance",
                   choices=["yfinance", "alphavantage", "binance"])
    p.add_argument("--indicators", nargs="+",
                   default=["RSI", "MACD"],
                   help="Indicator names (e.g. RSI MACD Bollinger)")
    p.add_argument("--blend",      default="weighted_sum",
                   choices=["weighted_sum", "voting", "regime_weighted", "phiai_chooses"],
                   help="Blend method")
    p.add_argument("--capital",    type=float, default=100_000.0, help="Initial capital")
    p.add_argument("--threshold",  type=float, default=0.15,  help="Signal threshold")
    p.add_argument("--position",   type=float, default=0.95,  help="Position size (0–1)")
    p.add_argument("--phiai",      action="store_true",       help="Run PhiAI optimisation")
    p.add_argument("--max-iter",   type=int,   default=20,    help="PhiAI max iterations")
    p.add_argument("--api-key",    default="",  help="Alpha Vantage API key (if needed)")
    p.add_argument("--no-cache",   action="store_true",       help="Bypass data cache")
    return p.parse_args()


def run_backtest_experiment(
    symbol: str = "SPY",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    tf: str = "1D",
    vendor: str = "yfinance",
    indicators: list[str] | None = None,
    blend: str = "weighted_sum",
    capital: float = 100_000.0,
    threshold: float = 0.15,
    position: float = 0.95,
    phiai: bool = False,
    max_iter: int = 20,
    api_key: str = "",
    log_trades_artifact: bool = False,
    trades_artifact_path: str = "artifacts/backtest_trades.csv",
    tracker: Any = None,
) -> dict[str, float]:
    from phinance.backtest import run_backtest
    from phinance.data import fetch_and_cache
    from phinance.optimization import run_phiai_optimization

    selected_indicators = indicators or ["RSI", "MACD"]
    if tracker is not None:
        tracker.log_params(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "tf": tf,
                "vendor": vendor,
                "blend": blend,
                "capital": capital,
                "threshold": threshold,
                "position": position,
                "phiai": phiai,
                "max_iter": max_iter,
                "indicator_count": len(selected_indicators),
            }
        )

    df = fetch_and_cache(
        vendor=vendor,
        symbol=symbol,
        timeframe=tf,
        start=start,
        end=end,
        api_key=api_key or None,
    )

    indicators_cfg = {name: {"enabled": True, "auto_tune": phiai, "params": {}} for name in selected_indicators}

    if phiai:
        indicators_cfg, _ = run_phiai_optimization(
            ohlcv=df,
            indicators=indicators_cfg,
            max_iter_per_indicator=max_iter,
            timeframe=tf,
        )

    result = run_backtest(
        ohlcv=df,
        symbol=symbol,
        indicators=indicators_cfg,
        blend_method=blend,
        signal_threshold=threshold,
        initial_capital=capital,
        position_size_pct=position,
    )

    if tracker is not None and log_trades_artifact:
        import pandas as pd

        trade_rows = [
            {
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "symbol": trade.symbol,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "hold_bars": trade.hold_bars,
                "direction": trade.direction,
                "regime": trade.regime,
            }
            for trade in result.trades
        ]
        artifact_path = Path(trades_artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trade_rows).to_csv(artifact_path, index=False)
        tracker.log_artifact(str(artifact_path))

    return {
        "total_return": float(result.total_return),
        "cagr": float(result.cagr),
        "max_drawdown": float(result.max_drawdown),
        "sharpe": float(result.sharpe),
        "sortino": float(result.sortino),
        "win_rate": float(result.win_rate),
        "total_trades": float(result.total_trades),
        "net_pl": float(result.net_pl),
    }


def run_experiment_target(
    symbol: str = "SPY",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    tf: str = "1D",
    vendor: str = "yfinance",
    indicators: list[str] | None = None,
    blend: str = "weighted_sum",
    capital: float = 100_000.0,
    threshold: float = 0.15,
    position: float = 0.95,
    phiai: bool = False,
    max_iter: int = 20,
    api_key: str = "",
    log_trades_artifact: bool = False,
    trades_artifact_path: str = "artifacts/backtest_trades.csv",
    tracker: Any = None,
) -> dict[str, float]:
    return run_backtest_experiment(
        symbol=symbol,
        start=start,
        end=end,
        tf=tf,
        vendor=vendor,
        indicators=indicators,
        blend=blend,
        capital=capital,
        threshold=threshold,
        position=position,
        phiai=phiai,
        max_iter=max_iter,
        api_key=api_key,
        log_trades_artifact=log_trades_artifact,
        trades_artifact_path=trades_artifact_path,
        tracker=tracker,
    )


def main() -> int:
    args = parse_args()

    from phinance.data import fetch_and_cache
    from phinance.backtest import run_backtest
    from phinance.storage import RunHistory
    from phinance.config.run_config import RunConfig

    print(f"\n{'='*60}")
    print(f"  Phi-nance CLI Backtest Runner")
    print(f"{'='*60}")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Period    : {args.start} → {args.end}")
    print(f"  Timeframe : {args.tf}")
    print(f"  Vendor    : {args.vendor}")
    print(f"  Indicators: {', '.join(args.indicators)}")
    print(f"  Blend     : {args.blend}")
    print(f"  Capital   : ${args.capital:,.0f}")
    print(f"  PhiAI     : {'on' if args.phiai else 'off'}")
    print(f"{'='*60}\n")

    # ── Fetch data ────────────────────────────────────────────────────────
    print("⬇  Fetching data…")
    df = fetch_and_cache(
        vendor    = args.vendor,
        symbol    = args.symbol,
        timeframe = args.tf,
        start     = args.start,
        end       = args.end,
        api_key   = args.api_key or None,
    )
    print(f"   Loaded {len(df):,} bars  [{df.index[0].date()} → {df.index[-1].date()}]\n")

    # ── Indicator config ──────────────────────────────────────────────────
    indicators = {
        name: {"enabled": True, "auto_tune": args.phiai, "params": {}}
        for name in args.indicators
    }

    # ── PhiAI optimisation ────────────────────────────────────────────────
    if args.phiai:
        from phinance.optimization import run_phiai_optimization
        print(f"🤖 PhiAI optimising {len(indicators)} indicator(s) "
              f"({args.max_iter} iter each)…")
        indicators, explanation = run_phiai_optimization(
            ohlcv                  = df,
            indicators             = indicators,
            max_iter_per_indicator = args.max_iter,
            timeframe              = args.tf,
        )
        print(f"\n{explanation}\n")

    # ── Run backtest ──────────────────────────────────────────────────────
    print("🚀 Running backtest…")
    result = run_backtest(
        ohlcv            = df,
        symbol           = args.symbol,
        indicators       = indicators,
        blend_method     = args.blend,
        signal_threshold = args.threshold,
        initial_capital  = args.capital,
        position_size_pct= args.position,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  Results for {args.symbol}")
    print(f"{'─'*40}")
    print(f"  Total Return  : {result.total_return:+.2%}")
    print(f"  CAGR          : {result.cagr:+.2%}")
    print(f"  Max Drawdown  : -{result.max_drawdown:.2%}")
    print(f"  Sharpe Ratio  : {result.sharpe:.3f}")
    print(f"  Sortino Ratio : {result.sortino:.3f}")
    print(f"  Win Rate      : {result.win_rate:.1%}")
    print(f"  Total Trades  : {result.total_trades}")
    print(f"  Net P&L       : ${result.net_pl:+,.2f}")
    print(f"{'─'*40}\n")

    # ── Persist run ───────────────────────────────────────────────────────
    cfg = RunConfig(
        symbols         = [args.symbol],
        start_date      = args.start,
        end_date        = args.end,
        timeframe       = args.tf,
        vendor          = args.vendor,
        initial_capital = args.capital,
        indicators      = indicators,
        blend_method    = args.blend,
        phiai_enabled   = args.phiai,
    )
    history = RunHistory()
    run_id  = history.create_run(cfg)
    history.save_results(run_id, result.to_dict())
    print(f"💾 Run saved:  {run_id}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
