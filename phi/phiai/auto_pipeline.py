"""
Fully Automated Pipeline — PhiAI + optional Ollama

One-shot: fetch data → select indicators → tune params → pick blend → ready for backtest.
No manual tuning. Uses Ollama when available for indicator/blend suggestions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Default indicators to use when Ollama unavailable
_DEFAULT_INDICATORS = ["RSI", "MACD", "Bollinger", "Dual SMA"]
_DEFAULT_PARAMS = {
    "RSI": {"rsi_period": 14, "oversold": 30, "overbought": 70},
    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "Bollinger": {"bb_period": 20, "num_std": 2},
    "Dual SMA": {"fast_period": 10, "slow_period": 50},
    "Mean Reversion": {"sma_period": 20},
    "Breakout": {"channel_period": 20},
    "Buy & Hold": {},
    "VWAP": {"band_pct": 0.5},
}

# Intraday-tuned defaults — shorter periods for 10-min to 10-hour signal horizon
_INTRADAY_TF = {"1m", "5m", "15m", "30m", "1H"}
_INTRADAY_DEFAULT_INDICATORS = ["RSI", "MACD", "VWAP", "Bollinger"]
_INTRADAY_DEFAULT_PARAMS = {
    "RSI": {"rsi_period": 7, "oversold": 30, "overbought": 70},
    "MACD": {"fast_period": 5, "slow_period": 13, "signal_period": 5},
    "Bollinger": {"bb_period": 14, "num_std": 2},
    "Dual SMA": {"fast_period": 5, "slow_period": 20},
    "Mean Reversion": {"sma_period": 10},
    "Breakout": {"channel_period": 10},
    "VWAP": {"band_pct": 0.3},
    "Buy & Hold": {},
}


def _ollama_suggest_indicators(
    symbol: str,
    start: str,
    end: str,
    host: str = "http://localhost:11434",
    model: str = "llama3.2",
) -> Optional[Tuple[List[str], str]]:
    """
    Ask Ollama which indicators to use. Returns (indicator_names, blend_method) or None.
    """
    try:
        from phi.agents import OllamaAgent, check_ollama_ready

        if not check_ollama_ready(host):
            return None

        agent = OllamaAgent(model=model, host=host)
        prompt = f"""For backtesting {symbol} from {start} to {end}, choose 2-4 indicators from: RSI, MACD, Bollinger, Dual SMA, Mean Reversion, Breakout, VWAP, Buy & Hold.
Pick blend: weighted_sum, regime_weighted, or voting.
Reply ONLY with valid JSON, no other text: {{"indicators": ["RSI","MACD"], "blend": "weighted_sum"}}"""

        reply = agent.chat(prompt, system="You are a quant. Reply only with JSON.")
        # Extract JSON from reply
        start = reply.find("{")
        if start >= 0:
            depth, end = 0, start
            for i, c in enumerate(reply[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            try:
                data = json.loads(reply[start : end + 1])
                inds = data.get("indicators", _DEFAULT_INDICATORS[:2])
                blend = data.get("blend", "weighted_sum")
                valid = {"RSI", "MACD", "Bollinger", "Dual SMA", "Mean Reversion", "Breakout", "VWAP", "Buy & Hold"}
                inds = [i for i in inds if i in valid][:4]
                if not inds:
                    inds = ["RSI", "MACD"]
                return inds, blend
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return None


def run_fully_automated(
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "1D",
    initial_capital: float = 100_000,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
    use_ollama: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict], str, str, pd.DataFrame]:
    """
    Fully automated pipeline: data → indicators → params → blend.

    Returns
    -------
    (config, indicators, blend_method, explanation, ohlcv)
    """
    from phi.data import fetch_and_cache
    from phi.phiai import run_phiai_optimization

    explanation_parts = []
    is_intraday = timeframe in _INTRADAY_TF

    # 1. Fetch data — use yfinance for intraday (no rate limiting), alphavantage for daily
    vendor = "yfinance" if is_intraday else "alphavantage"
    ohlcv = fetch_and_cache(vendor, symbol, timeframe, start_date, end_date)
    if ohlcv is None or len(ohlcv) < 20:
        raise ValueError(f"Could not load data for {symbol} ({timeframe})")
    explanation_parts.append(f"Data: {vendor} {symbol} {timeframe} ({len(ohlcv)} bars)")

    # 2. Select indicators (Ollama or timeframe-aware defaults)
    indicators_to_use = []
    blend_method = "weighted_sum"

    if use_ollama:
        sugg = _ollama_suggest_indicators(symbol, start_date, end_date, ollama_host, ollama_model)
        if sugg:
            indicators_to_use, blend_method = sugg
            explanation_parts.append(f"Ollama selected: {', '.join(indicators_to_use)}, blend={blend_method}")

    if not indicators_to_use:
        if is_intraday:
            indicators_to_use = _INTRADAY_DEFAULT_INDICATORS[:3]
        else:
            indicators_to_use = _DEFAULT_INDICATORS[:3]
        explanation_parts.append(f"{'Intraday' if is_intraday else 'Daily'} defaults: {', '.join(indicators_to_use)}")

    # 3. Build initial indicator configs with timeframe-appropriate params
    default_params = _INTRADAY_DEFAULT_PARAMS if is_intraday else _DEFAULT_PARAMS
    indicators = {}
    for name in indicators_to_use:
        params = default_params.get(name, {})
        indicators[name] = {"enabled": True, "auto_tune": True, "params": params.copy()}

    # 4. PhiAI optimize params using timeframe-aware grids
    optimized, opt_expl = run_phiai_optimization(ohlcv, indicators, max_iter_per_indicator=12, timeframe=timeframe)
    indicators = optimized
    explanation_parts.append(opt_expl)

    # 5. Blend weights (equal)
    blend_weights = {k: 1.0 / len(indicators) for k in indicators}

    from datetime import datetime

    # Ensure date strings have full format
    sd = start_date[:10] if len(start_date) >= 10 else start_date
    ed = end_date[:10] if len(end_date) >= 10 else end_date

    config = {
        "symbols": [symbol],
        "start": datetime.fromisoformat(sd + "T00:00:00"),
        "end": datetime.fromisoformat(ed + "T23:59:59"),
        "timeframe": timeframe,
        "vendor": vendor,
        "initial_capital": initial_capital,
        "benchmark": symbol,
        "trading_mode": "equities",
    }

    full_explanation = "\n".join(explanation_parts)
    return config, indicators, blend_method, full_explanation, ohlcv
