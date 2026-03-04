"""
app_streamlit/pages/live_trading_dashboard.py
==============================================

Streamlit page: Live / Paper Trading Dashboard
Provides real-time monitoring, order entry, position tracking,
and P&L visualization for the PaperBroker (or real broker adapters).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from phinance.live.broker_base import OrderSide, OrderType
from phinance.live.paper_engine import PaperBroker


# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_session():
    if "live_broker" not in st.session_state:
        broker = PaperBroker(initial_capital=100_000, slippage=0.0005, commission=0.01)
        broker.connect()
        st.session_state["live_broker"]    = broker
        st.session_state["equity_history"] = []
        st.session_state["pnl_history"]    = []
        st.session_state["last_prices"]    = {}


def _get_broker() -> PaperBroker:
    return st.session_state["live_broker"]


# ── Chart builders ─────────────────────────────────────────────────────────────

def _equity_chart(history: List[Dict]) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve", template="plotly_dark",
                          height=280, margin=dict(l=40, r=20, t=40, b=30))
        return fig

    df   = pd.DataFrame(history)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["equity"],
        mode="lines",
        line=dict(color="#a855f7", width=2),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)",
        name="Equity",
    ))
    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Time",
        yaxis_title="USD",
        showlegend=False,
    )
    return fig


def _positions_table(broker: PaperBroker, prices: Dict[str, float]) -> pd.DataFrame:
    positions = broker.get_positions()
    if not positions:
        return pd.DataFrame(columns=["Symbol", "Qty", "Avg Entry", "Current", "P&L", "P&L %"])
    rows = []
    for pos in positions:
        curr = prices.get(pos.symbol, pos.avg_entry)
        pnl  = pos.qty * (curr - pos.avg_entry)
        pct  = (curr / pos.avg_entry - 1) * 100 if pos.avg_entry else 0.0
        rows.append({
            "Symbol":    pos.symbol,
            "Qty":       pos.qty,
            "Avg Entry": f"${pos.avg_entry:.2f}",
            "Current":   f"${curr:.2f}",
            "P&L":       f"${pnl:+.2f}",
            "P&L %":     f"{pct:+.2f}%",
        })
    return pd.DataFrame(rows)


# ── Main page ─────────────────────────────────────────────────────────────────

def render_live_trading_dashboard():
    """Render the Live Trading Dashboard page."""
    _init_session()
    broker = _get_broker()

    st.title("📡 Live / Paper Trading Dashboard")
    st.caption("Real-time paper trading simulation with PaperBroker.")

    # ── Top controls ──────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        symbol = st.text_input("Symbol", value="SPY", key="lt_symbol").upper()
    with ctrl_col2:
        price = st.number_input("Set Price ($)", min_value=0.01, value=450.0,
                                step=0.5, key="lt_price")
        if st.button("Update Price", key="lt_update_price"):
            broker.update_price(symbol, float(price))
            st.session_state["last_prices"][symbol] = float(price)
            st.success(f"Price for {symbol} set to ${price:.2f}")
    with ctrl_col3:
        if st.button("🔄 Reset Broker", key="lt_reset"):
            broker.reset()
            st.session_state["equity_history"] = []
            st.session_state["pnl_history"]    = []
            st.session_state["last_prices"]    = {}
            st.success("Broker reset to $100,000 cash.")

    st.divider()

    # ── Account summary ───────────────────────────────────────────────────────
    acct = broker.get_account()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity",    f"${acct.get('equity', 0):,.2f}")
    m2.metric("Cash",      f"${acct.get('cash', 0):,.2f}")
    initial = acct.get("initial_capital", 100_000)
    total_pnl = acct.get("equity", initial) - initial
    m3.metric("Total P&L", f"${total_pnl:+,.2f}",
              delta=f"{total_pnl/initial*100:+.2f}%")
    m4.metric("Positions", acct.get("num_positions", len(broker.get_positions())))

    # Record equity
    st.session_state["equity_history"].append({
        "ts":     datetime.utcnow().strftime("%H:%M:%S"),
        "equity": acct.get("equity", initial),
    })
    # Keep only last 200 points
    if len(st.session_state["equity_history"]) > 200:
        st.session_state["equity_history"] = st.session_state["equity_history"][-200:]

    # ── Equity chart ──────────────────────────────────────────────────────────
    st.plotly_chart(
        _equity_chart(st.session_state["equity_history"]),
        use_container_width=True,
    )

    st.divider()

    # ── Order entry ───────────────────────────────────────────────────────────
    st.subheader("📋 Order Entry")
    oe_col1, oe_col2, oe_col3, oe_col4 = st.columns(4)
    with oe_col1:
        order_symbol = st.text_input("Symbol", value=symbol, key="oe_symbol").upper()
    with oe_col2:
        order_qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="oe_qty")
    with oe_col3:
        order_side = st.selectbox("Side", ["BUY", "SELL"], key="oe_side")
    with oe_col4:
        order_type = st.selectbox("Type", ["MARKET", "LIMIT"], key="oe_type")

    limit_price = None
    if order_type == "LIMIT":
        limit_price = st.number_input("Limit Price ($)", min_value=0.01,
                                       value=float(price), key="oe_limit")

    if st.button("🚀 Submit Order", key="lt_submit_order"):
        side      = OrderSide.BUY if order_side == "BUY" else OrderSide.SELL
        otype     = OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT
        cur_price = st.session_state["last_prices"].get(order_symbol, price)
        broker.update_price(order_symbol, float(cur_price))
        try:
            order = broker.submit_order(order_symbol, order_qty, side, otype,
                                        limit_price=limit_price)
            if order and order.is_filled:
                st.success(
                    f"✅ Order filled: {order_side} {order_qty} {order_symbol} "
                    f"@ ${order.filled_avg_price:.2f}"
                )
            else:
                st.info(f"Order submitted (status: {order.status if order else 'unknown'})")
        except Exception as exc:
            st.error(f"❌ Order failed: {exc}")

    st.divider()

    # ── Positions ─────────────────────────────────────────────────────────────
    st.subheader("📊 Open Positions")
    pos_df = _positions_table(broker, st.session_state["last_prices"])
    if pos_df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    # ── Recent fills ──────────────────────────────────────────────────────────
    st.subheader("🧾 Recent Fills")
    fills = broker.get_fills()
    if not fills:
        st.info("No fills yet.")
    else:
        fill_rows = [
            {
                "Symbol":  f.symbol,
                "Side":    f.side,
                "Qty":     f.qty,
                "Price":   f"${f.price:.2f}",
                "Notional":f"${f.qty * f.price:,.2f}",
                "Time":    str(f.filled_at)[:19] if f.filled_at else "—",
            }
            for f in reversed(fills[-20:])
        ]
        st.dataframe(pd.DataFrame(fill_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "⚠️ Paper trading only — no real capital at risk. "
        "Connect AlpacaBroker or IBKRBroker for live trading."
    )


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    render_live_trading_dashboard()
