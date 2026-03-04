"""
Phi-nance Backtest Agent
========================

Autonomously runs every strategy with multiple parameter sets, ranks the
results, sends them to Claude for analysis, and persists the winning
configuration so the live trading agent can load it on startup.

Usage (programmatic):
    from phi.agents import BacktestAgent
    agent = BacktestAgent()
    result = agent.run("SPY", "2022-01-01", "2024-12-31", capital=100_000)
    print(result["ai_analysis"])
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_LEARNED_DIR = _ROOT / "data_cache" / "learned_params"

# ─────────────────────────────────────────────────────────────────────────────
# Strategy catalogue — mirrors INDICATOR_CATALOG in live_workbench.py
# Each entry has defaults + up to two cheap variants
# ─────────────────────────────────────────────────────────────────────────────
_CATALOG: Dict[str, Dict] = {
    "RSI": {
        "module_cls": "strategies.rsi.RSIStrategy",
        "defaults": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "variants": [
            {"rsi_period": 10, "oversold": 25, "overbought": 75},
            {"rsi_period": 20, "oversold": 35, "overbought": 65},
        ],
    },
    "MACD": {
        "module_cls": "strategies.macd.MACDStrategy",
        "defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "variants": [
            {"fast_period": 8, "slow_period": 21, "signal_period": 5},
        ],
    },
    "Bollinger": {
        "module_cls": "strategies.bollinger.BollingerBands",
        "defaults": {"bb_period": 20, "num_std": 2.0},
        "variants": [
            {"bb_period": 15, "num_std": 1.5},
            {"bb_period": 25, "num_std": 2.5},
        ],
    },
    "Dual SMA": {
        "module_cls": "strategies.dual_sma.DualSMACrossover",
        "defaults": {"fast_period": 10, "slow_period": 50},
        "variants": [
            {"fast_period": 20, "slow_period": 100},
        ],
    },
    "Mean Reversion": {
        "module_cls": "strategies.mean_reversion.MeanReversion",
        "defaults": {"sma_period": 20},
        "variants": [
            {"sma_period": 10},
            {"sma_period": 50},
        ],
    },
    "Breakout": {
        "module_cls": "strategies.breakout.ChannelBreakout",
        "defaults": {"channel_period": 20},
        "variants": [
            {"channel_period": 10},
            {"channel_period": 40},
        ],
    },
    "Buy & Hold": {
        "module_cls": "strategies.buy_and_hold.BuyAndHold",
        "defaults": {},
        "variants": [],  # no point varying a pure benchmark
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_strategy(module_cls: str):
    module_path, cls_name = module_cls.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _scalar(val: Any, default: float = 0.0) -> float:
    """Unwrap lumibot's occasional dict-wrapping of metric values."""
    if isinstance(val, dict):
        val = next(iter(val.values()), default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _score(sharpe: float, cagr: float, max_dd: float) -> float:
    """Composite ranking score. Higher is better."""
    cagr_pct = cagr * 100
    dd_pct = abs(max_dd * 100)
    if cagr_pct < 0:
        return cagr_pct * 2          # heavily penalise losses
    return sharpe * 0.5 + cagr_pct * 0.3 - dd_pct * 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Single-run executor (Streamlit-free)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(
    strategy_cls,
    params: Dict,
    symbol: str,
    start: datetime,
    end: datetime,
    capital: float,
    df: pd.DataFrame,
    timeframe: str = "1D",
) -> Dict:
    """Execute one backtest and return a metrics dict."""
    try:
        from lumibot.backtesting import PandasDataBacktesting
        from lumibot.entities import Asset

        timestep = "day" if timeframe == "1D" else "minute"
        asset = Asset(symbol, asset_type=Asset.AssetTypes.STOCK)

        _df = df.copy()
        _df.columns = [c.lower() for c in _df.columns]
        _df.index = pd.to_datetime(_df.index)
        if _df.index.tz is not None:
            _df.index = _df.index.tz_localize(None)
        _df = _df.sort_index()

        pandas_data = {asset: {"df": _df, "timestep": timestep}}

        results, _strat = strategy_cls.run_backtest(
            datasource_class=PandasDataBacktesting,
            backtesting_start=start,
            backtesting_end=end,
            budget=capital,
            parameters=params,
            pandas_data=pandas_data,
            benchmark_asset=symbol,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
            show_indicators=False,
            show_progress_bar=False,
            quiet_logs=True,
        )

        sharpe = _scalar(results.get("sharpe_ratio"))
        cagr = _scalar(results.get("cagr"))
        max_dd = _scalar(results.get("max_drawdown"))
        total_return = _scalar(results.get("total_return"))

        return {
            "status": "ok",
            "sharpe": sharpe,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "total_return": total_return,
            "score": _score(sharpe, cagr, max_dd),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Main agent class
# ─────────────────────────────────────────────────────────────────────────────

class BacktestAgent:
    """
    Autonomous backtest optimizer + AI analyst.

    Call run() to kick off a full agent cycle.  Pass on_progress to receive
    status updates that can be rendered in any UI:

        on_progress(label: str, status: str, metrics: dict | None)

        status values: "fetching" | "running" | "complete" | "error"
    """

    def __init__(self) -> None:
        _LEARNED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        capital: float = 100_000,
        timeframe: str = "1D",
        on_progress: Optional[Callable] = None,
    ) -> Dict:
        """
        Full agent cycle.

        Returns:
            {
                "runs":              list of per-run metric dicts,
                "best":              the highest-scoring run,
                "ai_analysis":       Claude's markdown analysis (str),
                "ai_params":         Claude's recommended config (dict),
                "learned_params_path": path to the saved JSON file,
            }
        """
        symbol = symbol.upper()
        _emit = on_progress or (lambda *a, **k: None)

        # ── 1. Fetch data ─────────────────────────────────────────────────────
        _emit("Data", "fetching")
        from phi.data import auto_fetch_and_cache
        df, vendor_used = auto_fetch_and_cache(symbol, timeframe, start, end)
        _emit("Data", "complete", {"vendor": vendor_used, "bars": len(df)})

        start_dt = datetime.combine(date.fromisoformat(start), datetime.min.time())
        end_dt   = datetime.combine(date.fromisoformat(end),   datetime.min.time())

        # ── 2. Run all strategies (default + top variants) ────────────────────
        runs: List[Dict] = []

        for strat_name, cfg in _CATALOG.items():
            params_list = [cfg["defaults"]]
            if strat_name != "Buy & Hold":
                params_list.extend(cfg["variants"][:1])  # one variant each

            for i, raw_params in enumerate(params_list):
                label = strat_name if i == 0 else f"{strat_name} v{i+1}"
                _emit(label, "running")
                try:
                    strategy_cls = _load_strategy(cfg["module_cls"])
                except Exception as exc:
                    runs.append({"name": label, "strategy": cfg["module_cls"],
                                 "params": raw_params, "status": "error",
                                 "error": f"import failed: {exc}"})
                    _emit(label, "error", {"error": str(exc)})
                    continue

                full_params = {**raw_params, "symbol": symbol}
                metrics = _run_single(
                    strategy_cls, full_params,
                    symbol, start_dt, end_dt, capital, df, timeframe,
                )
                metrics.update({"name": label, "strategy": cfg["module_cls"],
                                 "params": raw_params})
                runs.append(metrics)
                _emit(label, "complete" if metrics["status"] == "ok" else "error", metrics)

        # ── 3. Rank results ───────────────────────────────────────────────────
        ok_runs = [r for r in runs if r["status"] == "ok"]
        ok_runs.sort(key=lambda r: r.get("score", -999), reverse=True)
        best = ok_runs[0] if ok_runs else None

        # ── 4. AI analysis via Claude ─────────────────────────────────────────
        _emit("Claude Analysis", "running")
        ai_analysis, ai_params = self._analyze_with_claude(
            symbol, start, end, runs, best
        )
        _emit("Claude Analysis", "complete")

        # ── 5. Persist learned params ─────────────────────────────────────────
        learned_path = self._save_learned_params(
            symbol, timeframe, start, end, best, ai_analysis, ai_params, ok_runs
        )

        return {
            "runs": runs,
            "best": best,
            "ai_analysis": ai_analysis,
            "ai_params": ai_params,
            "learned_params_path": str(learned_path),
        }

    def load_learned_params(self, symbol: str, timeframe: str = "1D") -> Optional[Dict]:
        """Load previously persisted params for a symbol/timeframe."""
        path = _LEARNED_DIR / f"{symbol.upper()}_{timeframe}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _analyze_with_claude(
        self,
        symbol: str,
        start: str,
        end: str,
        runs: List[Dict],
        best: Optional[Dict],
    ) -> Tuple[str, Dict]:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return self._fallback_analysis(runs, best), {}

        try:
            import anthropic

            rows = ["| Strategy | Sharpe | CAGR | Max DD | Return |",
                    "|----------|--------|------|--------|--------|"]
            for r in sorted(runs, key=lambda x: x.get("score", -999), reverse=True):
                if r["status"] == "ok":
                    rows.append(
                        f"| {r['name']} "
                        f"| {r['sharpe']:.2f} "
                        f"| {r['cagr']*100:.1f}% "
                        f"| {r['max_drawdown']*100:.1f}% "
                        f"| {r['total_return']*100:.1f}% |"
                    )
                else:
                    rows.append(f"| {r['name']} | — | — | — | ERROR |")

            table = "\n".join(rows)
            prompt = f"""\
You are a quantitative trading analyst reviewing automated backtest results.

**Symbol:** {symbol}  **Period:** {start} → {end}  **Runs:** {len(runs)}

{table}

Reply in this exact structure:

## What Worked
[2–3 sentences: best performers and why they likely fit this period]

## What Didn't Work
[1–2 sentences on underperformers]

## Market Conditions Inferred
[1–2 sentences on what regime this period appears to be: trending, ranging, volatile, etc.]

## Recommended Configuration
```json
{{
  "primary_strategy": "<name>",
  "params": {{}},
  "blend_secondary": "<name or null>",
  "blend_weights": {{"primary": 0.6, "secondary": 0.4}},
  "signal_threshold": 0.2,
  "rationale": "<one sentence>"
}}
```

## Lessons for Live Trading
- <lesson 1>
- <lesson 2>
- <lesson 3>"""

            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            return text, self._extract_json(text)

        except Exception as exc:
            fallback = self._fallback_analysis(runs, best)
            return f"{fallback}\n\n*Claude unavailable: {exc}*", {}

    @staticmethod
    def _fallback_analysis(runs: List[Dict], best: Optional[Dict]) -> str:
        if not best:
            return "No successful backtest runs completed."
        ok = sorted(
            [r for r in runs if r["status"] == "ok"],
            key=lambda r: r.get("score", 0), reverse=True,
        )
        lines = [
            f"**Best:** {best['name']} — "
            f"Sharpe {best['sharpe']:.2f}, "
            f"CAGR {best['cagr']*100:.1f}%, "
            f"MaxDD {best['max_drawdown']*100:.1f}%\n",
            "**All results (ranked):**",
        ]
        for r in ok:
            lines.append(
                f"- {r['name']}: Sharpe {r['sharpe']:.2f}, "
                f"CAGR {r['cagr']*100:.1f}%, "
                f"MaxDD {r['max_drawdown']*100:.1f}%"
            )
        lines.append("\n*Add ANTHROPIC_API_KEY to .env for AI-powered insights.*")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> Dict:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    def _save_learned_params(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        best: Optional[Dict],
        ai_analysis: str,
        ai_params: Dict,
        ok_runs: List[Dict],
    ) -> Path:
        path = _LEARNED_DIR / f"{symbol}_{timeframe}.json"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "backtest_period": {"start": start, "end": end},
            "best_run": {
                "name": best["name"] if best else None,
                "strategy": best["strategy"] if best else None,
                "params": best["params"] if best else {},
                "performance": {
                    "sharpe":        best["sharpe"]        if best else 0,
                    "cagr":          best["cagr"]          if best else 0,
                    "max_drawdown":  best["max_drawdown"]  if best else 0,
                    "total_return":  best["total_return"]  if best else 0,
                },
            },
            "ai_recommended": ai_params,
            "ai_analysis_summary": ai_analysis[:800],
            "all_runs": [
                {
                    "name":         r["name"],
                    "sharpe":       r.get("sharpe", 0),
                    "cagr":         r.get("cagr", 0),
                    "max_drawdown": r.get("max_drawdown", 0),
                    "params":       r.get("params", {}),
                }
                for r in ok_runs
            ],
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path
