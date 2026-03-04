#!/usr/bin/env python3
"""
Phi-nance Strategy Lab — Compact production UI. Run from repo root: streamlit run scripts/app_v2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np

from phinence.gui.strategy_catalog import (
    STRATEGY_CATALOG,
    COMPOUNDING_STRATEGIES,
    TESTING_MODES,
    BROKERAGES,
)

st.set_page_config(
    page_title="Phi-nance Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "strategy_configs" not in st.session_state:
    st.session_state.strategy_configs = {}
if "compounding_config" not in st.session_state:
    st.session_state.compounding_config = {"method": "majority", "params": {}}
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = []
if "regime_analysis" not in st.session_state:
    st.session_state.regime_analysis = {}
if "testing_mode" not in st.session_state:
    st.session_state.testing_mode = "phi_mode"
if "trade_config" not in st.session_state:
    st.session_state.trade_config = {
        "initial_balance": 10000,
        "commission_pct": 0.1,
        "commission_per_trade": 0.0,
        "brokerage": "tradier",
        "pdt_effect": False,
    }

# Dark mode: purple-to-orange theme (config in .streamlit/config.toml)
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; background: linear-gradient(180deg, #0f0614 0%, #1a0a1f 50%, #160d14 100%); }
.main .block-container { padding-top: 1.25rem; }
h1 { font-size: 1.5rem !important; font-weight: 600 !important; margin-bottom: 0.25rem !important; background: linear-gradient(90deg, #a855f7, #e8792a) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; }
h2 { font-size: 1rem !important; font-weight: 600 !important; margin-top: 0.75rem !important; margin-bottom: 0.35rem !important; color: #c4b5fd !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; color: #ddd6fe !important; }
p, .stCaption, [data-testid="stCaptionContainer"] { font-size: 0.8rem !important; color: #b8a9d4 !important; }
label { font-size: 0.8rem !important; color: #ddd6fe !important; }
.streamlit-expanderHeader { font-size: 0.9rem !important; background: linear-gradient(90deg, rgba(232,121,42,0.15), transparent) !important; border-left: 3px solid #e8792a !important; }
[data-testid="stExpander"] { background: rgba(26,15,36,0.8) !important; border: 1px solid rgba(168,85,247,0.3) !important; border-radius: 8px !important; }
.strat-row { padding: 0.4rem 0.6rem; margin: 0.2rem 0; border-radius: 6px; background: rgba(58,32,80,0.4); border: 1px solid rgba(168,85,247,0.25); }
.strat-row:hover { background: rgba(88,52,120,0.4); }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #e8792a !important; }
[data-testid="stMetricLabel"] { color: #b8a9d4 !important; }
hr { margin: 0.5rem 0 !important; border-color: rgba(168,85,247,0.35) !important; }
.dataframe { font-size: 0.8rem !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #7c3aed, #e8792a) !important; color: #0f0614 !important; border: none !important; font-weight: 600 !important; border-radius: 6px !important; }
.stButton > button[kind="primary"]:hover { opacity: 0.95 !important; box-shadow: 0 0 12px rgba(232,121,42,0.4) !important; }
.stButton > button { border-radius: 6px !important; border: 1px solid rgba(168,85,247,0.4) !important; color: #e8e0f0 !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { background: rgba(26,15,36,0.9) !important; border-color: rgba(168,85,247,0.35) !important; color: #e8e0f0 !important; }
[data-testid="stSelectbox"] { color: #e8e0f0 !important; }
[data-testid="stRadio"] label { color: #e8e0f0 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #e8792a) !important; }
</style>
""", unsafe_allow_html=True)

st.title("Phi-nance Strategy Lab")
st.caption("Select strategies, set compounding & test window, then run backtests. Phi mode = technical metrics; Trade mode = full simulation.")

st.subheader("Strategies")
for strategy in STRATEGY_CATALOG:
    if strategy["id"] not in st.session_state.strategy_configs:
        st.session_state.strategy_configs[strategy["id"]] = {
            "enabled": False,
            "params": {k: v["default"] for k, v in strategy.get("params", {}).items()}
        }

enabled_strategies = []
for strategy in STRATEGY_CATALOG:
    config = st.session_state.strategy_configs[strategy["id"]]
    with st.expander(f"{strategy['name']} — {strategy['category']}", expanded=False):
        enabled = st.checkbox("Enable this strategy", value=config["enabled"], key=f"enable_{strategy['id']}")
        config["enabled"] = enabled
        if strategy.get("params"):
            st.caption("Parameters")
            params_cols = st.columns(min(len(strategy["params"]), 4))
            for param_idx, (param_id, param_def) in enumerate(strategy["params"].items()):
                with params_cols[param_idx % len(params_cols)]:
                    if param_def["type"] == "int":
                        value = st.number_input(param_def["label"], min_value=param_def.get("min", 1), max_value=param_def.get("max", 1000), value=config["params"].get(param_id, param_def["default"]), key=f"param_{strategy['id']}_{param_id}")
                    elif param_def["type"] == "float":
                        value = st.number_input(param_def["label"], min_value=float(param_def.get("min", 0.0)), max_value=float(param_def.get("max", 1000.0)), value=float(config["params"].get(param_id, param_def["default"])), step=0.1, key=f"param_{strategy['id']}_{param_id}")
                    elif param_def["type"] == "bool":
                        value = st.checkbox(param_def["label"], value=config["params"].get(param_id, param_def["default"]), key=f"param_{strategy['id']}_{param_id}")
                    else:
                        value = st.text_input(param_def["label"], value=str(config["params"].get(param_id, param_def["default"])), key=f"param_{strategy['id']}_{param_id}")
                    config["params"][param_id] = value
    if config["enabled"]:
        enabled_strategies.append(strategy["id"])

if not enabled_strategies:
    st.caption("Enable at least one strategy above to run.")

st.subheader("Compounding")
if len(enabled_strategies) > 1:
    c1, c2 = st.columns([2, 4])
    with c1:
        compounding_method = st.selectbox("Method", options=[cs["id"] for cs in COMPOUNDING_STRATEGIES], format_func=lambda x: next((cs["name"] for cs in COMPOUNDING_STRATEGIES if cs["id"] == x), x), index=0, key="compounding_method")
    st.session_state.compounding_config["method"] = compounding_method
    selected_compounding = next(cs for cs in COMPOUNDING_STRATEGIES if cs["id"] == compounding_method)
    if selected_compounding.get("params"):
        with st.expander("Compounding parameters", expanded=False):
            comp_cols = st.columns(min(len(selected_compounding["params"]), 4))
            for comp_idx, (param_id, param_def) in enumerate(selected_compounding["params"].items()):
                with comp_cols[comp_idx % len(comp_cols)]:
                    if param_def["type"] == "int":
                        value = st.number_input(param_def["label"], min_value=param_def.get("min", 1), max_value=param_def.get("max", 1000), value=st.session_state.compounding_config["params"].get(param_id, param_def["default"]), key=f"comp_param_{param_id}")
                    elif param_def["type"] == "float":
                        value = st.number_input(param_def["label"], min_value=float(param_def.get("min", 0.0)), max_value=float(param_def.get("max", 1000.0)), value=float(st.session_state.compounding_config["params"].get(param_id, param_def["default"])), step=0.1, key=f"comp_param_{param_id}")
                    elif param_def["type"] == "bool":
                        value = st.checkbox(param_def["label"], value=st.session_state.compounding_config["params"].get(param_id, param_def["default"]), key=f"comp_param_{param_id}")
                    else:
                        value = st.text_input(param_def["label"], value=str(st.session_state.compounding_config["params"].get(param_id, param_def["default"])), key=f"comp_param_{param_id}")
                    st.session_state.compounding_config["params"][param_id] = value
else:
    st.caption("Enable 2+ strategies to set compounding.")

st.subheader("Test")
testing_mode = st.radio("Mode", options=[m["id"] for m in TESTING_MODES], format_func=lambda x: next((m["name"] for m in TESTING_MODES if m["id"] == x), x), horizontal=True, key="testing_mode_radio")
st.session_state.testing_mode = testing_mode
if testing_mode == "trade_mode":
    with st.expander("Trade settings", expanded=False):
        tc = st.session_state.trade_config
        r = st.columns(5)
        with r[0]:
            tc["initial_balance"] = st.number_input("Balance ($)", min_value=500, max_value=10_000_000, value=int(tc.get("initial_balance", 10000)), step=1000, key="trade_initial_balance")
        with r[1]:
            tc["commission_pct"] = st.number_input("Commission %", min_value=0.0, max_value=5.0, value=float(tc.get("commission_pct", 0.1)), step=0.01, key="trade_commission_pct")
        with r[2]:
            tc["commission_per_trade"] = st.number_input("Per trade ($)", min_value=0.0, max_value=100.0, value=float(tc.get("commission_per_trade", 0.0)), step=0.5, key="trade_commission_fixed")
        with r[3]:
            brokerage = st.selectbox("Brokerage", options=[b["id"] for b in BROKERAGES], format_func=lambda x: next((b["name"] for b in BROKERAGES if b["id"] == x), x), key="trade_brokerage")
            tc["brokerage"] = brokerage
        with r[4]:
            tc["pdt_effect"] = st.checkbox("PDT in effect", value=tc.get("pdt_effect", False), key="trade_pdt")
        st.session_state.trade_config = tc

cols = st.columns(5)
with cols[0]:
    ticker = st.text_input("Symbol", value="SPY", placeholder="SPY").strip() or "SPY"
with cols[1]:
    start = st.text_input("Start", value="2024-01-01", placeholder="YYYY-MM-DD")
with cols[2]:
    end = st.text_input("End", value="2024-06-30", placeholder="YYYY-MM-DD")
with cols[3]:
    run_mode = st.selectbox("Run", ["Compare individually", "Combine strategies"], key="run_mode")
with cols[4]:
    run_button = st.button("Run backtest", type="primary", use_container_width=True)

if run_button:
    if not enabled_strategies:
        st.warning("Please enable at least one strategy.")
    else:
        try:
            from phinence.gui.runner import run_backtest_for_strategy, run_combined_backtest
        except ImportError:
            st.error("Could not import runner. Ensure phinence is installed.")
            st.stop()
        results = []
        progress = st.progress(0, text="Running backtests...")
        testing_mode = st.session_state.testing_mode
        trade_config = st.session_state.trade_config
        strategy_params_map = {sid: st.session_state.strategy_configs[sid].get("params", {}) for sid in enabled_strategies if sid in st.session_state.strategy_configs}
        comp_config = st.session_state.compounding_config
        if run_mode == "Combine strategies" and len(enabled_strategies) > 1:
            progress.progress(0.5, text=f"Combined ({len(enabled_strategies)} strategies)...")
            r = run_combined_backtest(enabled_strategies, ticker, start, end, voting_mode=comp_config["method"], data_root=REPO_ROOT / "data" / "bars", strategy_params_map=strategy_params_map, compounding_params=comp_config.get("params", {}), testing_mode=testing_mode, trade_config=trade_config if testing_mode == "trade_mode" else None)
            results.append(r)
        else:
            for i, strategy_id in enumerate(enabled_strategies):
                progress.progress((i + 1) / len(enabled_strategies), text=f"Running {strategy_id}...")
                params = strategy_params_map.get(strategy_id, {})
                r = run_backtest_for_strategy(strategy_id, ticker, start, end, data_root=REPO_ROOT / "data" / "bars", strategy_params=params, testing_mode=testing_mode, trade_config=trade_config if testing_mode == "trade_mode" else None)
                results.append(r)
        progress.empty()
        st.session_state.backtest_results = results
        try:
            from phinence.gui.runner import detect_regime
            st.session_state.regime_analysis = detect_regime(ticker, start, end, data_root=REPO_ROOT / "data" / "bars")
        except Exception:
            st.session_state.regime_analysis = {"best_regime": "Unknown", "regime_distribution": {"Trending": 0.33, "Mean Reverting": 0.33, "Expanding": 0.34}}
        st.rerun()

st.subheader("Results")
if st.session_state.backtest_results:
    df = pd.DataFrame(st.session_state.backtest_results)
    testing_mode = st.session_state.get("testing_mode", "trade_mode")
    if "Error" in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        if "Sharpe Ratio" in df.columns and len(df) > 0:
            best_idx = df["Sharpe Ratio"].idxmax()
            best_name = df.loc[best_idx, "Strategy"]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Best strategy", best_name, f"Sharpe {df.loc[best_idx, 'Sharpe Ratio']:.2f}")
            with m2:
                if st.session_state.regime_analysis:
                    st.metric("Regime", st.session_state.regime_analysis.get("best_regime", "—"), "")
            with m3:
                if testing_mode == "phi_mode":
                    st.caption("Phi mode: technical metrics")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.session_state.regime_analysis and "regime_distribution" in st.session_state.regime_analysis:
            with st.expander("Regime distribution", expanded=False):
                for regime, pct in st.session_state.regime_analysis["regime_distribution"].items():
                    st.progress(pct, text=f"{regime}: {pct:.0%}")
else:
    st.caption("Enable strategies and run backtest to see results.")

with st.expander("Ask the agent", expanded=False):
    agent_input = st.chat_input("Ask about strategies or results...")
    if agent_input:
        if "agent_messages" not in st.session_state:
            st.session_state.agent_messages = []
        st.session_state.agent_messages.append({"role": "user", "content": agent_input})
        summary = str(st.session_state.backtest_results) if st.session_state.backtest_results else ""
        reply = "I'm here to help with your strategy analysis."
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                r = client.chat.completions.create(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "system", "content": "You are a trading strategy assistant."}, {"role": "user", "content": f"Backtest results: {summary}\n\nQuestion: {agent_input}"}], max_tokens=400)
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
