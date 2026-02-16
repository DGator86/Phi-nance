"""
Strategy catalog with parameter definitions for the GUI.
"""
from __future__ import annotations

from typing import Any

STRATEGY_CATALOG = [
    {
        "id": "buy_and_hold",
        "name": "Buy & Hold",
        "description": "Buy once and hold. Simple baseline strategy.",
        "category": "Baseline",
        "params": {},
    },
    {
        "id": "sma_cross",
        "name": "SMA Crossover",
        "description": "Buy when short SMA crosses above long SMA; sell when it crosses below.",
        "category": "Trend Following",
        "params": {
            "fast_period": {"label": "Fast SMA Period", "type": "int", "default": 10, "min": 2, "max": 100},
            "slow_period": {"label": "Slow SMA Period", "type": "int", "default": 20, "min": 5, "max": 200},
        },
    },
    {
        "id": "rsi",
        "name": "RSI Strategy",
        "description": "Buy when RSI < oversold threshold; sell when RSI > overbought threshold.",
        "category": "Mean Reversion",
        "params": {
            "rsi_period": {"label": "RSI Period", "type": "int", "default": 14, "min": 2, "max": 50},
            "oversold": {"label": "Oversold Threshold", "type": "float", "default": 30.0, "min": 10.0, "max": 50.0},
            "overbought": {"label": "Overbought Threshold", "type": "float", "default": 70.0, "min": 50.0, "max": 95.0},
        },
    },
    {
        "id": "bollinger",
        "name": "Bollinger Bands",
        "description": "Buy when price touches lower band; sell when price touches upper band.",
        "category": "Mean Reversion",
        "params": {
            "bb_period": {"label": "BB Period", "type": "int", "default": 20, "min": 5, "max": 100},
            "num_std": {"label": "Standard Deviations", "type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
        },
    },
    {
        "id": "macd",
        "name": "MACD",
        "description": "Buy on bullish MACD/signal crossover; sell on bearish crossover.",
        "category": "Trend Following",
        "params": {
            "fast_period": {"label": "Fast EMA", "type": "int", "default": 12, "min": 2, "max": 50},
            "slow_period": {"label": "Slow EMA", "type": "int", "default": 26, "min": 10, "max": 100},
            "signal_period": {"label": "Signal EMA", "type": "int", "default": 9, "min": 2, "max": 30},
        },
    },
    {
        "id": "momentum",
        "name": "Momentum",
        "description": "Buy when price momentum is positive; sell when negative.",
        "category": "Trend Following",
        "params": {
            "lookback": {"label": "Lookback Period", "type": "int", "default": 20, "min": 5, "max": 200},
        },
    },
    {
        "id": "mean_reversion",
        "name": "Mean Reversion",
        "description": "Buy when price < SMA (oversold); sell when price > SMA (overbought).",
        "category": "Mean Reversion",
        "params": {
            "sma_period": {"label": "SMA Period", "type": "int", "default": 20, "min": 5, "max": 200},
        },
    },
    {
        "id": "projection",
        "name": "Phi-nance Projection",
        "description": "Uses market projection (liquidity, regime, sentiment) for daily direction.",
        "category": "Advanced",
        "params": {},
    },
]

COMPOUNDING_STRATEGIES = [
    {
        "id": "majority",
        "name": "Majority Vote",
        "description": "Buy/sell when more than half of strategies agree.",
        "params": {
            "threshold": {"label": "Minimum Agreement %", "type": "float", "default": 50.0, "min": 0.0, "max": 100.0},
        },
    },
    {
        "id": "unanimous",
        "name": "Unanimous",
        "description": "Buy/sell only when all strategies agree.",
        "params": {},
    },
    {
        "id": "weighted",
        "name": "Weighted Average",
        "description": "Weight signals by strategy performance or custom weights.",
        "params": {
            "buy_threshold": {"label": "Buy Signal Threshold", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "sell_threshold": {"label": "Sell Signal Threshold", "type": "float", "default": -0.3, "min": -1.0, "max": 0.0},
            "use_performance_weights": {"label": "Use Performance Weights", "type": "bool", "default": False},
        },
    },
    {
        "id": "at_least_n",
        "name": "At Least N Agree",
        "description": "Buy/sell when at least N strategies agree.",
        "params": {
            "min_agreement": {"label": "Minimum Strategies", "type": "int", "default": 2, "min": 1, "max": 10},
        },
    },
]
