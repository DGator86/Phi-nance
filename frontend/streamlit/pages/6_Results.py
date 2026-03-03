"""
Page 6 — Results
=================

Display backtest results: KPI cards, equity curve, trade table, Phibot review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Results | Phi-nance", layout="wide")
st.title("6 · Results")
st.caption("Backtest performance analysis and Phibot AI review.")

result = st.session_state.get("backtest_result")
run_id = st.session_state.get("run_id")

if result is None:
    st.warning("No backtest results yet. Run a backtest in Step 5.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return",  f"{result.total_return:.1%}")
c2.metric("CAGR",          f"{result.cagr:.1%}")
c3.metric("Max Drawdown",  f"-{result.max_drawdown:.1%}")
c4.metric("Sharpe Ratio",  f"{result.sharpe:.2f}")
c5.metric("Total Trades",  str(result.total_trades))

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────
if result.portfolio_value:
    import plotly.graph_objects as go

    ohlcv  = st.session_state.get("ohlcv")
    symbol = st.session_state.get("symbol", "")

    # Align PV to index length
    pv = result.portfolio_value
    if ohlcv is not None and len(pv) == len(ohlcv) + 1:
        pv = pv[1:]  # drop the synthetic initial value
    idx = ohlcv.index[:len(pv)] if ohlcv is not None else list(range(len(pv)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=pv,
        name="Portfolio Value",
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
    ))
    initial_cap = st.session_state.get("initial_capital", 100_000)
    fig.add_hline(y=initial_cap, line_dash="dot", line_color="#64748b", annotation_text="Initial")
    fig.update_layout(
        title="Equity Curve",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Trade table ───────────────────────────────────────────────────────────────
if result.trades:
    with st.expander(f"📋 Trade Log ({len(result.trades)} trades)", expanded=False):
        trade_rows = [
            {
                "Entry":  str(t.entry_date)[:10] if t.entry_date else "",
                "Exit":   str(t.exit_date)[:10]  if t.exit_date  else "",
                "Price In": f"${t.entry_price:.2f}",
                "Price Out":f"${t.exit_price:.2f}",
                "Qty":     t.quantity,
                "P&L":    f"${t.pnl:+.2f}",
                "P&L %":  f"{t.pnl_pct:+.1%}",
                "Win":    "✅" if t.win else "❌",
                "Regime": t.regime,
            }
            for t in result.trades
        ]
        df_trades = pd.DataFrame(trade_rows)
        st.dataframe(df_trades, use_container_width=True)

# ── Phibot AI Review ──────────────────────────────────────────────────────────
st.subheader("🤖 Phibot AI Review")

if st.button("Generate Phibot Review", type="primary"):
    with st.spinner("Phibot is analysing your backtest…"):
        try:
            from phinance.phibot.reviewer import review_backtest

            ohlcv      = st.session_state.get("ohlcv")
            indicators = st.session_state.get("indicators", {})
            weights    = st.session_state.get("blend_weights", {})
            blend_meth = st.session_state.get("blend_method", "weighted_sum")

            review = review_backtest(
                ohlcv          = ohlcv,
                results        = result.to_dict(),
                prediction_log = result.prediction_log,
                indicators     = indicators,
                blend_weights  = weights,
                blend_method   = blend_meth,
                config         = {"symbols": [st.session_state.get("symbol", "")]},
            )

            # Verdict badge
            _VERDICT_COLORS = {
                "strong":   "#22c55e",
                "moderate": "#f59e0b",
                "weak":     "#f97316",
                "neutral":  "#94a3b8",
            }
            color = _VERDICT_COLORS.get(review.verdict, "#94a3b8")
            st.markdown(
                f'<div style="padding:1rem; border-left:4px solid {color}; '
                f'background: rgba(0,0,0,0.1); border-radius:4px">'
                f'<b>Verdict: {review.verdict.upper()}</b> — {review.summary}'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Observations
            st.subheader("Observations")
            for obs in review.observations:
                st.markdown(f"- {obs}")

            # Tweaks
            if review.tweaks:
                st.subheader("Recommended Tweaks")
                for tw in review.tweaks:
                    with st.expander(f"**[{tw.confidence.upper()}]** {tw.title}"):
                        st.markdown(tw.rationale)
                        col1, col2 = st.columns(2)
                        col1.metric("Current", str(tw.current_value))
                        col2.metric("Suggested", str(tw.suggested_value))

        except Exception as exc:
            st.error(f"Phibot review failed: {exc}")

# ── Run history summary ───────────────────────────────────────────────────────
with st.expander("📚 Recent Run History", expanded=False):
    try:
        from phinance.storage import RunHistory

        history = RunHistory()
        runs = history.list_runs(limit=20)
        if runs:
            hist_rows = [
                {
                    "Run ID":      r["run_id"],
                    "Symbol":      ", ".join(r["config"].get("symbols", [])),
                    "TF":          r["config"].get("timeframe", ""),
                    "Total Return":f"{r['results'].get('total_return', 0):.1%}",
                    "Sharpe":      f"{r['results'].get('sharpe', 0):.2f}",
                }
                for r in runs
            ]
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
        else:
            st.info("No previous runs stored.")
    except Exception as exc:
        st.warning(f"Could not load run history: {exc}")
