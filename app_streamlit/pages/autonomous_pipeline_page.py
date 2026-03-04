"""
app_streamlit/pages/autonomous_pipeline_page.py
================================================

Streamlit page: Autonomous Strategy Pipeline
One-click strategy proposal, validation, deployment, and rollback —
powered by StrategyProposerAgent, StrategyValidator, AutonomousDeployer.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tests.fixtures.ohlcv import make_ohlcv   # demo data only


# ── Lazy imports (avoid loading heavy deps until page opened) ─────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline_classes():
    from phinance.agents.strategy_proposer import StrategyProposerAgent
    from phinance.agents.strategy_validator import StrategyValidator
    from phinance.agents.autonomous_deployer import (
        AutonomousDeployer,
        DeploymentStatus,
    )
    return StrategyProposerAgent, StrategyValidator, AutonomousDeployer, DeploymentStatus


# ── Session helpers ───────────────────────────────────────────────────────────

def _init_session():
    if "ap_deployer" not in st.session_state:
        _, _, AutonomousDeployer, _ = _load_pipeline_classes()
        st.session_state["ap_deployer"]  = AutonomousDeployer(dry_run=True)
        st.session_state["ap_proposal"]  = None
        st.session_state["ap_vr"]        = None
        st.session_state["ap_rec"]       = None
        st.session_state["ap_history"]   = []   # list of run summaries


# ── Small chart ───────────────────────────────────────────────────────────────

def _mini_equity_chart(backtest_stats: dict) -> go.Figure:
    """Very minimal P&L spark-line from backtest stats dict."""
    fig = go.Figure()
    total_return = backtest_stats.get("total_return", 0.0)
    n = 50
    arr = np.linspace(0, total_return * 100, n) + np.random.default_rng(42).normal(0, 0.5, n)
    arr = np.cumsum(arr / n) + 100
    fig.add_trace(go.Scatter(
        y=arr,
        mode="lines",
        line=dict(
            color="#22c55e" if total_return >= 0 else "#ef4444",
            width=2,
        ),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.07)" if total_return >= 0 else "rgba(239,68,68,0.07)",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=120,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ── Metric helpers ────────────────────────────────────────────────────────────

def _colour(value: float, good_positive: bool = True) -> str:
    if (value >= 0) == good_positive:
        return "green"
    return "red"


def _pct(v: float) -> str:
    return f"{v*100:+.1f}%"


# ── Main page ─────────────────────────────────────────────────────────────────

def render_autonomous_pipeline():
    """Render the Autonomous Strategy Pipeline page."""
    _init_session()
    deployer = st.session_state["ap_deployer"]

    StrategyProposerAgent, StrategyValidator, AutonomousDeployer, DeploymentStatus = (
        _load_pipeline_classes()
    )

    st.title("🤖 Autonomous Strategy Pipeline")
    st.caption(
        "Propose → Validate → Deploy strategies without human intervention. "
        "All runs use synthetic demo data; connect your live data feed for production use."
    )

    # ── Configuration sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Pipeline Config")
        top_n = st.slider("Top-N indicators to propose", 1, 8, 3)
        n_bars = st.slider("Demo data bars", 100, 500, 300)
        min_sharpe   = st.number_input("Min Sharpe (validation)", value=-5.0, step=0.5)
        max_drawdown = st.number_input("Max Drawdown (validation)", value=1.0, min_value=0.0,
                                       max_value=1.0, step=0.05)
        min_trades   = st.number_input("Min Trades (validation)", value=0, min_value=0, step=1)
        dry_run      = st.toggle("Dry-run mode", value=True)
        if dry_run != deployer.dry_run:
            st.session_state["ap_deployer"] = AutonomousDeployer(dry_run=dry_run)
            deployer = st.session_state["ap_deployer"]

    # ── Demo data ─────────────────────────────────────────────────────────────
    df_demo = make_ohlcv(n=n_bars)

    # ── Step 1: Propose ───────────────────────────────────────────────────────
    st.subheader("Step 1 — Propose Strategy")
    if st.button("🔍 Run Proposer", key="ap_propose"):
        with st.spinner("Analysing regime and scoring indicators…"):
            t0     = time.time()
            agent  = StrategyProposerAgent(top_n=top_n)
            proposal = agent.propose(df_demo)
            elapsed  = (time.time() - t0) * 1000
            st.session_state["ap_proposal"] = proposal
            st.session_state["ap_vr"]  = None
            st.session_state["ap_rec"] = None

    proposal = st.session_state.get("ap_proposal")
    if proposal:
        with st.expander("📋 Proposal Details", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Regime Detected",  proposal.regime)
            c2.metric("Top Indicators",   str(list(proposal.indicators.keys())))
            c3.metric("Blend Method",     proposal.blend_method)
            st.write("**Weights:**", proposal.weights)
            st.write("**Scores:**",  proposal.scores)
            st.write("**Rationale:**", proposal.rationale)

    st.divider()

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    st.subheader("Step 2 — Validate Strategy")
    if proposal and st.button("✅ Run Validator", key="ap_validate"):
        with st.spinner("Running walk-forward backtest…"):
            validator = StrategyValidator(
                min_sharpe   = min_sharpe,
                max_drawdown = max_drawdown,
                min_win_rate = 0.0,
                min_trades   = int(min_trades),
            )
            vr = validator.validate(proposal, df_demo)
            st.session_state["ap_vr"]  = vr
            st.session_state["ap_rec"] = None

    vr = st.session_state.get("ap_vr")
    if vr:
        with st.expander("📊 Validation Results", expanded=True):
            approved_icon = "✅ APPROVED" if vr.approved else "❌ REJECTED"
            st.markdown(f"### {approved_icon}")
            if not vr.approved:
                st.error(f"Rejection reason: {vr.rejection_reason}")

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Sharpe Ratio",   f"{vr.sharpe:+.3f}")
            v2.metric("Max Drawdown",   f"{vr.max_drawdown*100:.1f}%")
            v3.metric("Win Rate",       f"{vr.win_rate*100:.1f}%")
            v4.metric("Total Return",   f"{vr.total_return*100:+.1f}%")

            st.metric("Num Trades", vr.num_trades)

            if vr.backtest_stats:
                st.plotly_chart(
                    _mini_equity_chart(vr.backtest_stats),
                    use_container_width=True,
                )

    st.divider()

    # ── Step 3: Deploy ────────────────────────────────────────────────────────
    st.subheader("Step 3 — Deploy Strategy")
    if vr and vr.approved and st.button("🚀 Deploy Strategy", key="ap_deploy"):
        custom_name = st.session_state.get("ap_custom_name", "AutoStrategy")
        with st.spinner("Deploying…"):
            try:
                rec = deployer.deploy(vr, strategy_name=custom_name)
                st.session_state["ap_rec"] = rec
                st.session_state["ap_history"].append({
                    "id":      rec.deployment_id[:8],
                    "name":    rec.strategy_name,
                    "status":  rec.status,
                    "regime":  rec.regime_at_deploy,
                    "sharpe":  rec.validation_stats.get("sharpe", 0.0),
                    "dry_run": rec.dry_run,
                })
                st.success(f"✅ Deployed! ID: `{rec.deployment_id}`")
            except Exception as exc:
                st.error(f"❌ Deployment failed: {exc}")
    elif vr and not vr.approved:
        st.warning("Strategy was rejected — fix validation thresholds or re-run proposer.")

    st.text_input("Strategy name (optional)", key="ap_custom_name",
                  value="AutoStrategy", label_visibility="collapsed")

    rec = st.session_state.get("ap_rec")
    if rec:
        with st.expander("📦 Deployment Record", expanded=True):
            d1, d2, d3 = st.columns(3)
            d1.metric("Status",   str(rec.status))
            d2.metric("Dry Run",  "Yes" if rec.dry_run else "NO — LIVE")
            d3.metric("Regime",   rec.regime_at_deploy)
            st.code(rec.deployment_id, language=None)

            if st.button("🔙 Rollback", key="ap_rollback"):
                ok = deployer.rollback(rec.deployment_id)
                if ok:
                    st.warning(f"Strategy `{rec.deployment_id[:8]}` rolled back.")
                    st.session_state["ap_rec"] = None
                else:
                    st.error("Rollback failed.")

    st.divider()

    # ── Active deployments ────────────────────────────────────────────────────
    st.subheader("🗂 Active Deployments")
    active = deployer.list_active()
    if not active:
        st.info("No active deployments.")
    else:
        rows = []
        for r in active:
            rows.append({
                "ID (short)":   r.deployment_id[:12],
                "Name":         r.strategy_name,
                "Status":       r.status,
                "Regime":       r.regime_at_deploy,
                "Dry Run":      "✅" if r.dry_run else "🔴 LIVE",
                "Sharpe":       f"{r.validation_stats.get('sharpe', 0.0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state["ap_history"]:
        with st.expander("📜 Session Deploy History"):
            st.dataframe(
                pd.DataFrame(st.session_state["ap_history"]),
                use_container_width=True,
                hide_index=True,
            )


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    render_autonomous_pipeline()
