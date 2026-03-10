"""Streamlit UI for limit-order-book simulations."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import pandas as pd
import streamlit as st

from phi.lob.data import iter_events, load_custom_csv, load_dukascopy_ticks, load_lobster_csv
from phi.lob.engine import LobSimEngine
from phi.lob.strategy import ImbalanceStrategy, MarketMakingStrategy
from phi.lob.synthetic import generate_synthetic_events


def _strategy_from_ui(name: str, params: dict[str, float]):
    if name == "Market Making":
        return MarketMakingStrategy(spread_bps=float(params["spread_bps"]), size=float(params["size"]))
    return ImbalanceStrategy(threshold=float(params["threshold"]), size=float(params["size"]))


def render_lob_dashboard() -> None:
    """Render controls and results for LOB simulation runs."""
    st.title("📚 Limit Order Book Simulation")

    source = st.selectbox("Data source", ["Synthetic", "LOBSTER", "Dukascopy", "Custom CSV"])
    max_events = st.slider("Events to process", min_value=50, max_value=5000, value=500, step=50)

    events_iter = None
    synthetic_params: dict[str, float] = {}
    if source == "Synthetic":
        c1, c2, c3 = st.columns(3)
        synthetic_params["arrival_rate"] = c1.number_input("Arrival rate", value=20.0, min_value=0.1)
        synthetic_params["volatility"] = c2.number_input("Volatility", value=0.02, min_value=0.0001)
        synthetic_params["spread_bps"] = c3.number_input("Spread (bps)", value=2.0, min_value=0.1)
        synthetic_params["seed"] = int(st.number_input("Random seed", value=42, min_value=0))
        events_iter = generate_synthetic_events(
            arrival_rate=float(synthetic_params["arrival_rate"]),
            volatility=float(synthetic_params["volatility"]),
            spread_bps=float(synthetic_params["spread_bps"]),
            seed=int(synthetic_params["seed"]),
        )
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            temp_path = Path("/tmp/lob_upload.csv")
            temp_path.write_bytes(uploaded.getvalue())
            if source == "LOBSTER":
                df = load_lobster_csv(temp_path)
            elif source == "Dukascopy":
                df = load_dukascopy_ticks(temp_path)
            else:
                st.caption("Map your columns to: timestamp, event_type, price, volume, side")
                column_map_raw = st.text_area(
                    "Column map (JSON)",
                    value='{"timestamp":"timestamp","event_type":"event_type","price":"price","volume":"volume","side":"side"}',
                )
                try:
                    col_map = json.loads(column_map_raw)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in column map.")
                    return
                df = load_custom_csv(temp_path, col_map)
            events_iter = iter_events(df)

    strategy_name = st.selectbox("Strategy", ["Market Making", "Order Flow Imbalance"])
    if strategy_name == "Market Making":
        strategy_params = {
            "spread_bps": st.number_input("MM Spread (bps)", value=3.0, min_value=0.1),
            "size": st.number_input("Order size", value=1.0, min_value=0.1),
        }
    else:
        strategy_params = {
            "threshold": st.slider("Imbalance threshold", min_value=0.01, max_value=0.95, value=0.25),
            "size": st.number_input("Order size", value=1.0, min_value=0.1),
        }

    if st.button("Run LOB Simulation", type="primary"):
        if events_iter is None:
            st.warning("Please provide a valid data source.")
            return

        strategy = _strategy_from_ui(strategy_name, strategy_params)
        capped_events = islice(events_iter, max_events)
        engine = LobSimEngine(capped_events, strategy)
        result = engine.run()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PnL", f"{result.metrics['pnl']:.2f}")
        m2.metric("Return", f"{result.metrics['return']:.2%}")
        m3.metric("Sharpe", f"{result.metrics['sharpe']:.3f}")
        m4.metric("Trades", int(result.metrics["trades"]))

        st.subheader("Equity Curve")
        if result.equity_curve.empty:
            st.info("No equity records generated.")
        else:
            st.line_chart(result.equity_curve.set_index("timestamp")["equity"])

        st.subheader("Depth Snapshot")
        depth = engine.book.market_depth(levels=10)
        d1, d2 = st.columns(2)
        d1.dataframe(pd.DataFrame(depth["bids"], columns=["price", "volume"]), use_container_width=True)
        d2.dataframe(pd.DataFrame(depth["asks"], columns=["price", "volume"]), use_container_width=True)

        st.subheader("Trade Log")
        if result.fills:
            fills_df = pd.DataFrame(
                [{"timestamp": fill.timestamp, "side": fill.side.value, "qty": fill.quantity, "price": fill.price} for fill in result.fills]
            )
            st.dataframe(fills_df, use_container_width=True)
        else:
            st.info("No fills produced.")
