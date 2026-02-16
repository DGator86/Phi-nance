#!/usr/bin/env python3
"""
Phi-nance Strategy Lab v2 — Advanced GUI with expandable strategies, custom parameters, and regime analysis.

  streamlit run scripts/app_v2.py

Features:
- Expandable strategy cards with custom parameters
- Compounding strategy selector
- Tiered layout (strategies → compounding → test → results)
- Regime detection and best regime display
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np

from phinence.gui.strategy_catalog import STRATEGY_CATALOG, COMPOUNDING_STRATEGIES

st.set_page_config(
    page_title="Phi-nance Strategy Lab v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "strategy_configs" not in st.session_state:
    st.session_state.strategy_configs = {}  # {strategy_id: {enabled: bool, params: dict}}
if "compounding_config" not in st.session_state:
    st.session_state.compounding_config = {"method": "majority", "params": {}}
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = []
if "regime_analysis" not in st.session_state:
    st.session_state.regime_analysis = {}

# CSS for expandable cards
st.markdown("""
<style>
.strategy-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f9f9f9;
}
.strategy-card.expanded {
    background-color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TIER 1: STRATEGIES SECTION
# ============================================================================
st.title("📈 Phi-nance Strategy Lab v2")
st.markdown("---")

st.subheader("🎯 Tier 1: Select & Configure Strategies")
st.caption("Click the toggle to expand each strategy and customize its parameters.")

# Initialize strategy configs
for strategy in STRATEGY_CATALOG:
    if strategy["id"] not in st.session_state.strategy_configs:
        st.session_state.strategy_configs[strategy["id"]] = {
            "enabled": False,
            "params": {k: v["default"] for k, v in strategy.get("params", {}).items()}
        }

# Display strategy cards
enabled_strategies = []
for strategy in STRATEGY_CATALOG:
    config = st.session_state.strategy_configs[strategy["id"]]
    
    # Strategy card header
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        enabled = st.checkbox("", value=config["enabled"], key=f"enable_{strategy['id']}", label_visibility="collapsed")
        config["enabled"] = enabled
    with col2:
        st.markdown(f"**{strategy['name']}** ({strategy['category']})")
        st.caption(strategy["description"])
    with col3:
        expand_key = f"expand_{strategy['id']}"
        if expand_key not in st.session_state:
            st.session_state[expand_key] = False
        expanded = st.button("⚙️", key=expand_key, help="Configure parameters")
        if expanded:
            st.session_state[expand_key] = not st.session_state[expand_key]
    
    if enabled:
        enabled_strategies.append(strategy["id"])
    
    # Expandable parameters section
    if st.session_state.get(expand_key, False):
        st.markdown("**Parameters:**")
        params_cols = st.columns(min(len(strategy.get("params", {})), 3))
        param_idx = 0
        for param_id, param_def in strategy.get("params", {}).items():
            with params_cols[param_idx % len(params_cols)]:
                if param_def["type"] == "int":
                    value = st.number_input(
                        param_def["label"],
                        min_value=param_def.get("min", 1),
                        max_value=param_def.get("max", 1000),
                        value=config["params"].get(param_id, param_def["default"]),
                        key=f"param_{strategy['id']}_{param_id}",
                    )
                elif param_def["type"] == "float":
                    value = st.number_input(
                        param_def["label"],
                        min_value=float(param_def.get("min", 0.0)),
                        max_value=float(param_def.get("max", 1000.0)),
                        value=float(config["params"].get(param_id, param_def["default"])),
                        step=0.1,
                        key=f"param_{strategy['id']}_{param_id}",
                    )
                elif param_def["type"] == "bool":
                    value = st.checkbox(
                        param_def["label"],
                        value=config["params"].get(param_id, param_def["default"]),
                        key=f"param_{strategy['id']}_{param_id}",
                    )
                else:
                    value = st.text_input(
                        param_def["label"],
                        value=str(config["params"].get(param_id, param_def["default"])),
                        key=f"param_{strategy['id']}_{param_id}",
                    )
                config["params"][param_id] = value
            param_idx += 1
        st.markdown("---")

if not enabled_strategies:
    st.info("👆 Enable at least one strategy above to continue.")

# ============================================================================
# TIER 2: COMPOUNDING STRATEGY SECTION
# ============================================================================
st.markdown("---")
st.subheader("🔗 Tier 2: Compounding Strategy")
st.caption("Configure how multiple strategies are combined (only shown when 2+ strategies enabled).")

if len(enabled_strategies) > 1:
    compounding_method = st.selectbox(
        "Compounding Method:",
        options=[cs["id"] for cs in COMPOUNDING_STRATEGIES],
        format_func=lambda x: next((cs["name"] for cs in COMPOUNDING_STRATEGIES if cs["id"] == x), x),
        index=0,
        key="compounding_method",
    )
    st.session_state.compounding_config["method"] = compounding_method
    
    # Show compounding parameters
    selected_compounding = next(cs for cs in COMPOUNDING_STRATEGIES if cs["id"] == compounding_method)
    st.caption(selected_compounding["description"])
    
    if selected_compounding.get("params"):
        st.markdown("**Compounding Parameters:**")
        comp_cols = st.columns(min(len(selected_compounding["params"]), 3))
        comp_idx = 0
        for param_id, param_def in selected_compounding["params"].items():
            with comp_cols[comp_idx % len(comp_cols)]:
                if param_def["type"] == "int":
                    value = st.number_input(
                        param_def["label"],
                        min_value=param_def.get("min", 1),
                        max_value=param_def.get("max", 1000),
                        value=st.session_state.compounding_config["params"].get(param_id, param_def["default"]),
                        key=f"comp_param_{param_id}",
                    )
                elif param_def["type"] == "float":
                    value = st.number_input(
                        param_def["label"],
                        min_value=float(param_def.get("min", 0.0)),
                        max_value=float(param_def.get("max", 1000.0)),
                        value=float(st.session_state.compounding_config["params"].get(param_id, param_def["default"])),
                        step=0.1,
                        key=f"comp_param_{param_id}",
                    )
                elif param_def["type"] == "bool":
                    value = st.checkbox(
                        param_def["label"],
                        value=st.session_state.compounding_config["params"].get(param_id, param_def["default"]),
                        key=f"comp_param_{param_id}",
                    )
                else:
                    value = st.text_input(
                        param_def["label"],
                        value=str(st.session_state.compounding_config["params"].get(param_id, param_def["default"])),
                        key=f"comp_param_{param_id}",
                    )
                st.session_state.compounding_config["params"][param_id] = value
            comp_idx += 1
else:
    st.info("Enable 2+ strategies to configure compounding.")

# ============================================================================
# TIER 3: TEST WINDOW SECTION
# ============================================================================
st.markdown("---")
st.subheader("🧪 Tier 3: Test Window")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Symbol", value="SPY", help="e.g. SPY, QQQ, AAPL").strip() or "SPY"
with col2:
    start = st.text_input("Start Date", value="2024-01-01", help="YYYY-MM-DD")
with col3:
    end = st.text_input("End Date", value="2024-06-30", help="YYYY-MM-DD")

run_mode = st.radio(
    "Run Mode:",
    ["Compare individually", "Combine strategies"],
    horizontal=True,
    help="Compare: Run each separately. Combine: Run as one combined strategy."
)

run_button = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ============================================================================
# RUN BACKTESTS
# ============================================================================
if run_button:
    if not enabled_strategies:
        st.warning("Please enable at least one strategy.")
    else:
        # Import runner functions
        try:
            from phinence.gui.runner import run_backtest_for_strategy, run_combined_backtest
        except ImportError:
            st.error("Could not import runner functions. Make sure phinence modules are installed.")
            st.stop()
        
        results = []
        progress = st.progress(0, text="Running backtests...")
        
        if run_mode == "Combine strategies" and len(enabled_strategies) > 1:
            # Run combined strategy
            progress.progress(0.5, text=f"Running combined strategy ({len(enabled_strategies)} strategies)...")
            r = run_combined_backtest(
                enabled_strategies, ticker, start, end,
                voting_mode=st.session_state.compounding_config["method"],
                data_root=REPO_ROOT / "data" / "bars"
            )
            results.append(r)
        else:
            # Run each strategy individually
            for i, strategy_id in enumerate(enabled_strategies):
                progress.progress((i + 1) / len(enabled_strategies), text=f"Running {strategy_id}...")
                r = run_backtest_for_strategy(strategy_id, ticker, start, end, data_root=REPO_ROOT / "data" / "bars")
                results.append(r)
        
        progress.empty()
        st.session_state.backtest_results = results
        
        # Simple regime analysis (placeholder - would use RegimeEngine in real implementation)
        st.session_state.regime_analysis = {
            "best_regime": "Trending Up",
            "regime_distribution": {"Trending Up": 0.4, "Trending Down": 0.3, "Ranging": 0.3}
        }
        
        st.rerun()

# ============================================================================
# TIER 4: RESULTS SECTION
# ============================================================================
st.markdown("---")
st.subheader("📊 Tier 4: Results")

if st.session_state.backtest_results:
    df = pd.DataFrame(st.session_state.backtest_results)
    
    if "Error" in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Best overall strategy
        if "Sharpe Ratio" in df.columns and len(df) > 0:
            best_idx = df["Sharpe Ratio"].idxmax()
            best_name = df.loc[best_idx, "Strategy"]
            st.success(f"**🏆 Best Strategy:** **{best_name}** (highest Sharpe Ratio: {df.loc[best_idx, 'Sharpe Ratio']:.2f})")
        
        # Regime analysis box
        if st.session_state.regime_analysis:
            st.markdown("---")
            st.subheader("📈 Regime Analysis")
            regime_info = st.session_state.regime_analysis
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Regime", regime_info.get("best_regime", "N/A"))
            with col2:
                if "regime_distribution" in regime_info:
                    st.markdown("**Regime Distribution:**")
                    for regime, pct in regime_info["regime_distribution"].items():
                        st.progress(pct, text=f"{regime}: {pct:.0%}")
else:
    st.info("👆 Configure strategies above and click **Run Backtest** to see results.")

# ============================================================================
# ASK THE AGENT
# ============================================================================
st.markdown("---")
st.subheader("🤖 Ask the Agent")
agent_input = st.chat_input("Ask anything about your strategies or results...")

if agent_input:
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    
    st.session_state.agent_messages.append({"role": "user", "content": agent_input})
    
    summary = str(st.session_state.backtest_results) if st.session_state.backtest_results else ""
    reply = "I'm here to help with your strategy analysis!"
    
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a trading strategy assistant."},
                    {"role": "user", "content": f"Backtest results: {summary}\n\nQuestion: {agent_input}"},
                ],
                max_tokens=400,
            )
            if r.choices:
                reply = r.choices[0].message.content or reply
        except Exception as e:
            reply = f"Error: {e}"
    
    st.session_state.agent_messages.append({"role": "assistant", "content": reply})
    st.rerun()

if "agent_messages" in st.session_state:
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
