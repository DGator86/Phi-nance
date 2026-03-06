"""Standalone live dashboard with data source health panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_live_dashboard(data_source_manager=None, broker=None, engine=None) -> None:
    st.title("📡 Live Trading Dashboard")

    if broker is not None:
        st.subheader("Account")
        try:
            account = broker.get_account()
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity", f"${account.get('equity', 0):,.2f}")
            c2.metric("Cash", f"${account.get('cash', 0):,.2f}")
            c3.metric("Positions", account.get("num_positions", 0))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Unable to load account: {exc}")

    if engine is not None:
        st.subheader("Advisor Reports")
        latest = getattr(engine, "last_advisor_report", None)
        if latest:
            st.info(latest)
        else:
            st.caption("No advisor report yet.")

        symbol = st.text_input("Explanation symbol", value="SPY")
        if st.button("Request trade explanation"):
            quote = {"symbol": symbol, "price": 0.0}
            report = engine.request_advisor_report(symbol=symbol, quote=quote)
            if report:
                st.success("Advisor response generated.")
                st.write(report)
            else:
                st.warning("Advisor not enabled or unavailable.")

    st.subheader("Data Sources")
    if data_source_manager is None:
        st.info("Data source manager not connected.")
        return

    snapshot = data_source_manager.usage_snapshot()
    rows = []
    for source, info in snapshot.items():
        rows.append({
            "source": source,
            "enabled": info["enabled"],
            "health": info["health"],
            "calls_made": info["calls_made"],
            "daily_used": info["daily_used"],
            "daily_limit": info["daily_limit"],
            "daily_remaining": info["daily_remaining"],
            "last_error": info["last_error"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Manual Controls")
    source_names = sorted(snapshot.keys())
    selected = st.selectbox("Source", source_names) if source_names else None
    if selected is not None:
        col1, col2 = st.columns(2)
        if col1.button("Enable source"):
            data_source_manager.set_source_enabled(selected, True)
            st.success(f"Enabled {selected}")
        if col2.button("Disable source"):
            data_source_manager.set_source_enabled(selected, False)
            st.warning(f"Disabled {selected}")


if __name__ == "__main__":
    render_live_dashboard()
