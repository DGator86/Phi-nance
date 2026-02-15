#!/usr/bin/env python3
"""
Phi-nance Backtesting Dashboard
--------------------------------
Interactive Streamlit UI for toggling strategies on/off, configuring
parameters, running backtests, and comparing **prediction accuracy**.

The primary success metric is how well each strategy predicts the
next-day price direction — not P&L.

Launch:
    streamlit run dashboard.py
"""

import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, date

import pandas as pd
import streamlit as st

from lumibot.backtesting import YahooDataBacktesting

from strategies.bollinger import BollingerBands
from strategies.breakout import ChannelBreakout
from strategies.buy_and_hold import BuyAndHold
from strategies.dual_sma import DualSMACrossover
from strategies.macd import MACDStrategy
from strategies.mean_reversion import MeanReversion
from strategies.momentum import MomentumRotation
from strategies.prediction_tracker import compute_prediction_accuracy
from strategies.rsi import RSIStrategy
from strategies.wyckoff import WyckoffStrategy

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here
# ---------------------------------------------------------------------------
STRATEGY_CATALOG = {
    "Buy & Hold": {
        "class": BuyAndHold,
        "description": (
            "Always predicts UP (permanent bull). "
            "Naive baseline every other strategy should beat."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        },
    },
    "Momentum Rotation": {
        "class": MomentumRotation,
        "description": (
            "Predicts UP for the strongest-momentum asset, DOWN for the rest."
        ),
        "params": {
            "symbols": {
                "label": "Universe (comma-separated)",
                "type": "text",
                "default": "SPY,VEU,AGG,GLD",
            },
            "lookback_days": {
                "label": "Lookback (days)",
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 200,
            },
            "rebalance_days": {
                "label": "Rebalance every (days)",
                "type": "number",
                "default": 5,
                "min": 1,
                "max": 60,
            },
        },
    },
    "Mean Reversion (SMA)": {
        "class": MeanReversion,
        "description": (
            "Predicts UP when price < SMA (oversold), "
            "DOWN when price > SMA (overbought)."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "sma_period": {
                "label": "SMA Period",
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 200,
            },
        },
    },
    "RSI": {
        "class": RSIStrategy,
        "description": (
            "Predicts UP when RSI < oversold (bounce expected), "
            "DOWN when RSI > overbought (pullback expected)."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "rsi_period": {
                "label": "RSI Period",
                "type": "number",
                "default": 14,
                "min": 2,
                "max": 50,
            },
            "oversold": {
                "label": "Oversold threshold",
                "type": "number",
                "default": 30,
                "min": 10,
                "max": 50,
            },
            "overbought": {
                "label": "Overbought threshold",
                "type": "number",
                "default": 70,
                "min": 50,
                "max": 95,
            },
        },
    },
    "Bollinger Bands": {
        "class": BollingerBands,
        "description": (
            "Predicts UP below lower band (oversold), "
            "DOWN above upper band (overbought)."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "bb_period": {
                "label": "BB Period",
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 100,
            },
            "num_std": {
                "label": "Std Deviations",
                "type": "number",
                "default": 2,
                "min": 1,
                "max": 4,
            },
        },
    },
    "MACD": {
        "class": MACDStrategy,
        "description": (
            "Predicts UP on bullish MACD/signal crossover, "
            "DOWN on bearish crossover."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "fast_period": {
                "label": "Fast EMA",
                "type": "number",
                "default": 12,
                "min": 2,
                "max": 50,
            },
            "slow_period": {
                "label": "Slow EMA",
                "type": "number",
                "default": 26,
                "min": 10,
                "max": 100,
            },
            "signal_period": {
                "label": "Signal EMA",
                "type": "number",
                "default": 9,
                "min": 2,
                "max": 30,
            },
        },
    },
    "Dual SMA Crossover": {
        "class": DualSMACrossover,
        "description": (
            "Golden cross (fast > slow) = UP, "
            "death cross (fast < slow) = DOWN. Trend-following."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "fast_period": {
                "label": "Fast SMA",
                "type": "number",
                "default": 10,
                "min": 2,
                "max": 100,
            },
            "slow_period": {
                "label": "Slow SMA",
                "type": "number",
                "default": 50,
                "min": 10,
                "max": 300,
            },
        },
    },
    "Channel Breakout": {
        "class": ChannelBreakout,
        "description": (
            "Donchian-style: predicts UP on new high breakout, "
            "DOWN on new low breakdown."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "channel_period": {
                "label": "Channel Period",
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 100,
            },
        },
    },
    "Wyckoff": {
        "class": WyckoffStrategy,
        "description": (
            "Accumulation (low in range + volume on up-days) = UP. "
            "Distribution (high in range + volume on down-days) = DOWN."
        ),
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "lookback": {
                "label": "Lookback (days)",
                "type": "number",
                "default": 30,
                "min": 10,
                "max": 120,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def build_sidebar():
    st.sidebar.title("Backtest Settings")

    start_date = st.sidebar.date_input(
        "Start date", value=date(2020, 1, 1), min_value=date(2005, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End date", value=date(2024, 12, 31), min_value=start_date
    )
    budget = st.sidebar.number_input(
        "Budget ($)", value=100_000, min_value=1_000, step=10_000
    )
    benchmark = st.sidebar.text_input("Benchmark symbol", value="SPY")

    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Lumibot + Yahoo Finance (free data)")

    return {
        "start": datetime.combine(start_date, datetime.min.time()),
        "end": datetime.combine(end_date, datetime.min.time()),
        "budget": float(budget),
        "benchmark": benchmark,
    }


# ---------------------------------------------------------------------------
# Strategy cards
# ---------------------------------------------------------------------------
def render_strategy_card(name, info):
    enabled = st.toggle(f"**{name}**", key=f"toggle_{name}")

    if enabled:
        st.caption(info["description"])
        params = {}
        for key, spec in info["params"].items():
            widget_key = f"{name}_{key}"
            if spec["type"] == "text":
                params[key] = st.text_input(
                    spec["label"], value=spec["default"], key=widget_key
                )
            elif spec["type"] == "number":
                params[key] = st.number_input(
                    spec["label"],
                    value=spec["default"],
                    min_value=spec.get("min", 1),
                    max_value=spec.get("max", 9999),
                    key=widget_key,
                )
        return True, params
    else:
        st.caption(info["description"])
        return False, None


def resolve_params(name, raw_params):
    resolved = dict(raw_params)
    if name == "Momentum Rotation" and "symbols" in resolved:
        resolved["symbols"] = [
            s.strip() for s in resolved["symbols"].split(",") if s.strip()
        ]
    int_keys = (
        "lookback_days", "rebalance_days", "sma_period", "rsi_period",
        "oversold", "overbought", "bb_period", "fast_period", "slow_period",
        "signal_period", "channel_period", "lookback",
    )
    for key in int_keys:
        if key in resolved:
            resolved[key] = int(resolved[key])
    # num_std stays float
    return resolved


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------
def run_single_backtest(strategy_class, params, config):
    results, strat = strategy_class.run_backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=config["start"],
        backtesting_end=config["end"],
        budget=config["budget"],
        benchmark_asset=config["benchmark"],
        parameters=params,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False,
        show_progress_bar=False,
        quiet_logs=True,
    )
    return results, strat


# ---------------------------------------------------------------------------
# Accuracy display helpers
# ---------------------------------------------------------------------------
def display_accuracy_metrics(name, scorecard):
    """Show prediction accuracy metrics for a single strategy."""
    if scorecard["total_predictions"] == 0:
        st.warning(f"{name}: no predictions recorded.")
        return

    # Top-level accuracy gauge
    acc = scorecard["accuracy"]
    total = scorecard["total_predictions"]
    hits = scorecard["hits"]
    misses = scorecard["misses"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.1%}")
    col2.metric("Predictions", f"{total:,}")
    col3.metric("Correct", f"{hits:,}")
    col4.metric("Wrong", f"{misses:,}")

    # Directional breakdown
    col_up, col_down, col_edge = st.columns(3)
    col_up.metric(
        "UP Accuracy",
        f"{scorecard['up_accuracy']:.1%}",
        help=f"{scorecard['up_predictions']} UP predictions total",
    )
    col_down.metric(
        "DOWN Accuracy",
        f"{scorecard['down_accuracy']:.1%}",
        help=f"{scorecard['down_predictions']} DOWN predictions total",
    )
    col_edge.metric(
        "Edge (avg magnitude)",
        f"${scorecard['edge']:.4f}",
        help=(
            f"Avg correct move: ${scorecard['avg_correct_magnitude']:.4f} | "
            f"Avg incorrect move: ${scorecard['avg_incorrect_magnitude']:.4f}"
        ),
    )

    # Rolling accuracy chart
    scored = scorecard["scored_log"]
    if len(scored) > 10:
        df = pd.DataFrame(scored)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["rolling_accuracy"] = (
            df["correct"].rolling(window=50, min_periods=10).mean()
        )
        chart_df = df[["date", "rolling_accuracy"]].dropna().set_index("date")
        st.line_chart(chart_df, y="rolling_accuracy", use_container_width=True)
        st.caption("50-day rolling prediction accuracy")

    # Prediction log (collapsed)
    with st.expander("View prediction log"):
        if scored:
            log_df = pd.DataFrame(scored)
            log_df["date"] = pd.to_datetime(log_df["date"]).dt.date
            log_df["actual_move"] = log_df["actual_move"].map(lambda x: f"${x:+.2f}")
            log_df["correct"] = log_df["correct"].map(
                lambda x: "Yes" if x else "No"
            )
            st.dataframe(
                log_df[
                    ["date", "symbol", "signal", "price", "next_price",
                     "actual_move", "correct"]
                ],
                use_container_width=True,
                hide_index=True,
            )


def display_comparison_table(all_scorecards):
    """Side-by-side comparison of all strategies by accuracy."""
    rows = []
    for name, sc in all_scorecards.items():
        if sc["total_predictions"] > 0:
            rows.append({
                "Strategy": name,
                "Accuracy": f"{sc['accuracy']:.1%}",
                "Predictions": sc["total_predictions"],
                "UP Acc.": f"{sc['up_accuracy']:.1%}",
                "DOWN Acc.": f"{sc['down_accuracy']:.1%}",
                "Edge": f"${sc['edge']:.4f}",
                "Correct": sc["hits"],
                "Wrong": sc["misses"],
            })

    if rows:
        # Sort by accuracy descending
        rows.sort(key=lambda r: float(r["Accuracy"].strip("%")) / 100, reverse=True)
        st.table(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Phi-nance Prediction Accuracy",
        page_icon="$",
        layout="wide",
    )

    st.title("Phi-nance Prediction Accuracy Dashboard")
    st.markdown(
        "Toggle strategies on/off, tweak parameters, then hit **Run Backtests**. "
        "Success is measured by **how accurately each strategy predicts "
        "next-day price direction** — not by P&L."
    )

    config = build_sidebar()

    # --- Strategy cards (rows of 4) ---
    st.header("Strategies")
    selected = {}
    catalog_items = list(STRATEGY_CATALOG.items())
    cols_per_row = 4

    for row_start in range(0, len(catalog_items), cols_per_row):
        row_items = catalog_items[row_start:row_start + cols_per_row]
        columns = st.columns(cols_per_row)
        for col, (name, info) in zip(columns, row_items):
            with col:
                with st.container(border=True):
                    enabled, params = render_strategy_card(name, info)
                    if enabled and params is not None:
                        selected[name] = {
                            "class": info["class"],
                            "params": resolve_params(name, params),
                        }

    # --- Run button ---
    st.markdown("---")

    if not selected:
        st.info("Enable at least one strategy above, then click Run Backtests.")
        return

    enabled_names = ", ".join(selected.keys())
    st.caption(f"Ready to run: **{enabled_names}**")

    if st.button("Run Backtests", type="primary", use_container_width=True):
        st.header("Prediction Accuracy Results")

        all_scorecards = {}
        for name, entry in selected.items():
            with st.status(f"Running {name}...", expanded=True) as status:
                st.write(
                    f"Backtesting **{name}** from "
                    f"{config['start'].date()} to {config['end'].date()}..."
                )
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        results, strat = run_single_backtest(
                            entry["class"], entry["params"], config
                        )
                    scorecard = compute_prediction_accuracy(strat)
                    all_scorecards[name] = scorecard
                    status.update(
                        label=f"{name} — {scorecard['accuracy']:.1%} accuracy",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"{name} — failed", state="error")
                    st.error(f"Error running {name}: {e}")

        # --- Per-strategy detail ---
        for name, scorecard in all_scorecards.items():
            st.subheader(name)
            display_accuracy_metrics(name, scorecard)

        # --- Comparison table ---
        if len(all_scorecards) > 1:
            st.markdown("---")
            st.subheader("Strategy Comparison (ranked by accuracy)")
            display_comparison_table(all_scorecards)


if __name__ == "__main__":
    main()
