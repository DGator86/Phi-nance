"""Live trading dashboard page."""

from __future__ import annotations

import streamlit as st

from phi.live.engine import LiveEngine


def render_live_dashboard() -> None:
    st.title("📡 Live Trading")
    if "live_engine" not in st.session_state:
        st.session_state.live_engine = None

    c1, c2 = st.columns(2)
    if c1.button("Start Engine"):
        engine = LiveEngine.from_settings()
        engine.run(max_cycles=1)
        st.session_state.live_engine = engine
    if c2.button("Stop Engine") and st.session_state.live_engine is not None:
        st.session_state.live_engine.stop()

    engine = st.session_state.live_engine
    if engine is None:
        st.info("Engine not started.")
        return

    acct = engine.broker.get_account()
    m1, m2, m3 = st.columns(3)
    m1.metric("Cash", f"${acct.cash:,.2f}")
    m2.metric("Equity", f"${acct.equity:,.2f}")
    m3.metric("Buying Power", f"${acct.buying_power:,.2f}")

    st.subheader("Positions")
    st.dataframe(engine.portfolio.snapshot_positions(), use_container_width=True)

    st.subheader("Open Orders")
    st.dataframe([o.__dict__ for o in engine.broker.get_open_orders()], use_container_width=True)

    st.subheader("Action Log")
    st.code("\n".join(engine.action_log[-50:]) or "No actions yet")

    st.subheader("Equity Curve")
    st.line_chart(engine.portfolio.equity_curve, x="timestamp", y="equity")
