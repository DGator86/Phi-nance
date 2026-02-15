#!/usr/bin/env python3
"""
Phi-nance Backtesting Dashboard
--------------------------------
Interactive Streamlit UI for toggling strategies on/off, configuring
parameters, running backtests, and comparing results side-by-side.

Launch:
    streamlit run dashboard.py
"""

import io
import sys
import threading
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, date

import streamlit as st

from lumibot.backtesting import YahooDataBacktesting

from strategies.buy_and_hold import BuyAndHold
from strategies.momentum import MomentumRotation
from strategies.mean_reversion import MeanReversion

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here
# ---------------------------------------------------------------------------
STRATEGY_CATALOG = {
    "Buy & Hold": {
        "class": BuyAndHold,
        "icon": "B",
        "description": "Buys a single asset on day 1 and holds for the entire period.",
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        },
    },
    "Momentum Rotation": {
        "class": MomentumRotation,
        "icon": "M",
        "description": "Rotates into whichever asset has the strongest recent momentum.",
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
        "icon": "R",
        "description": "Buys below the SMA, sells above — a classic mean-reversion signal.",
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
}


def build_sidebar():
    """Render sidebar with global settings and return config dict."""
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


def render_strategy_card(name, info):
    """Render a toggle card for one strategy. Returns (enabled, params) or (False, None)."""
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
    """Convert UI strings into the types each strategy expects."""
    resolved = dict(raw_params)
    if name == "Momentum Rotation" and "symbols" in resolved:
        resolved["symbols"] = [
            s.strip() for s in resolved["symbols"].split(",") if s.strip()
        ]
    for key in ("lookback_days", "rebalance_days", "sma_period"):
        if key in resolved:
            resolved[key] = int(resolved[key])
    return resolved


def run_single_backtest(strategy_class, params, config):
    """Run one strategy backtest and return (results, strategy_instance)."""
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


def display_results(name, results):
    """Show key metrics for a completed backtest."""
    if results is None:
        st.warning(f"{name}: backtest returned no results.")
        return

    metrics = {
        "Total Return": results.get("total_return"),
        "CAGR": results.get("cagr"),
        "Sharpe Ratio": results.get("sharpe"),
        "Max Drawdown": results.get("max_drawdown"),
        "Volatility": results.get("volatility"),
    }

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        if value is not None:
            if label in ("Total Return", "CAGR", "Max Drawdown", "Volatility"):
                col.metric(label, f"{value:,.2%}")
            else:
                col.metric(label, f"{value:,.2f}")
        else:
            col.metric(label, "N/A")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Phi-nance Backtester",
        page_icon="$",
        layout="wide",
    )

    st.title("Phi-nance Backtesting Dashboard")
    st.markdown("Toggle strategies on/off, tweak parameters, then hit **Run Backtests**.")

    config = build_sidebar()

    # --- Strategy cards ---
    st.header("Strategies")
    selected = {}
    columns = st.columns(len(STRATEGY_CATALOG))

    for col, (name, info) in zip(columns, STRATEGY_CATALOG.items()):
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
        st.header("Results")

        all_results = {}
        for name, entry in selected.items():
            with st.status(f"Running {name}...", expanded=True) as status:
                st.write(f"Backtesting **{name}** from "
                         f"{config['start'].date()} to {config['end'].date()} "
                         f"with ${config['budget']:,.0f} budget...")
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        results, strat = run_single_backtest(
                            entry["class"], entry["params"], config
                        )
                    all_results[name] = results
                    status.update(label=f"{name} — done!", state="complete")
                except Exception as e:
                    status.update(label=f"{name} — failed", state="error")
                    st.error(f"Error running {name}: {e}")

        # --- Per-strategy results ---
        for name, results in all_results.items():
            st.subheader(name)
            display_results(name, results)

        # --- Comparison table ---
        if len(all_results) > 1:
            st.markdown("---")
            st.subheader("Side-by-Side Comparison")

            rows = []
            for name, results in all_results.items():
                if results:
                    rows.append({
                        "Strategy": name,
                        "Total Return": f"{results.get('total_return', 0):,.2%}",
                        "CAGR": f"{results.get('cagr', 0):,.2%}",
                        "Sharpe": f"{results.get('sharpe', 0):,.2f}",
                        "Max Drawdown": f"{results.get('max_drawdown', 0):,.2%}",
                        "Volatility": f"{results.get('volatility', 0):,.2%}",
                    })

            if rows:
                st.table(rows)


if __name__ == "__main__":
    main()
