#!/usr/bin/env python3
"""
Phi-nance Strategy Lab v3 — Visual Workflow Builder
Drag-and-drop strategy blocks, connect them, configure parameters, and run backtests.

  streamlit run app_v3.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
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
import streamlit.components.v1 as components

from phinence.gui.strategy_catalog import (
    STRATEGY_CATALOG,
    COMPOUNDING_STRATEGIES,
    TESTING_MODES,
    BROKERAGES,
)

st.set_page_config(
    page_title="Phi-nance Strategy Lab v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = {
        "blocks": [],  # [{id, type, x, y, params, connections: [to_id]}]
        "connections": [],  # [{from_id, to_id}]
    }
if "backtest_config" not in st.session_state:
    st.session_state.backtest_config = {
        "ticker": "SPY",
        "backtest_type": "date_range",  # or "bars"
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "bars_count": 1000,
        "timeframe": "1m",
    }
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# Dark mode CSS
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; background: linear-gradient(180deg, #0f0614 0%, #1a0a1f 50%, #160d14 100%); }
h1 { font-size: 1.5rem !important; font-weight: 600 !important; background: linear-gradient(90deg, #a855f7, #e8792a) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
h2 { font-size: 1rem !important; font-weight: 600 !important; color: #c4b5fd !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #7c3aed, #e8792a) !important; color: #0f0614 !important; border: none !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Phi-nance Strategy Lab v3 — Visual Workflow Builder")

# Load the drag-and-drop component HTML
COMPONENT_DIR = REPO_ROOT / "components" / "workflow_builder"
HTML_FILE = COMPONENT_DIR / "workflow_builder.html"

# Prepare strategy data for JS
strategies_json = json.dumps([{"id": s["id"], "name": s["name"], "category": s["category"], "params": s.get("params", {})} for s in STRATEGY_CATALOG])

# Main layout: Workspace (left) + Strategy Store (right)
col_left, col_right = st.columns([3, 1])

with col_right:
    st.subheader("Strategy Store")
    st.caption("Click to add blocks")
    
    # Strategy catalog as clickable items
    for strategy in STRATEGY_CATALOG:
        if st.button(f"➕ {strategy['name']}", key=f"add_{strategy['id']}", use_container_width=True):
            # Add block to workflow
            block_id = f"block_{len(st.session_state.workflow['blocks'])}"
            st.session_state.workflow["blocks"].append({
                "id": block_id,
                "type": strategy["id"],
                "x": 50 + (len(st.session_state.workflow["blocks"]) % 3) * 150,
                "y": 50 + (len(st.session_state.workflow["blocks"]) // 3) * 120,
                "params": {k: v["default"] for k, v in strategy.get("params", {}).items()}
            })
            st.rerun()
    
    # Composer block
    if st.button("🎼 Composer", key="add_composer", use_container_width=True):
        block_id = f"block_{len(st.session_state.workflow['blocks'])}"
        st.session_state.workflow["blocks"].append({
            "id": block_id,
            "type": "composer",
            "x": 50 + (len(st.session_state.workflow["blocks"]) % 3) * 150,
            "y": 50 + (len(st.session_state.workflow["blocks"]) // 3) * 120,
            "params": {"method": "majority"}
        })
        st.rerun()

with col_left:
    st.subheader("Workspace")
    
    # Enhanced HTML component with data injection
    if HTML_FILE.exists():
        html_content = HTML_FILE.read_text(encoding="utf-8")
        # Inject strategy data and current workflow state
        html_content = html_content.replace(
            'window.strategies = [];',
            f'window.strategies = {strategies_json};'
        )
        workflow_json = json.dumps(st.session_state.workflow)
        html_content = html_content.replace(
            '</script>',
            f'init({workflow_json});</script>'
        )
        
        # Use components.html with height
        result = components.html(html_content, height=600)
        
        # Handle updates from component (if using proper Streamlit component API)
        # For now, we'll use a simpler approach with buttons
    else:
        # Fallback: visual grid layout using Streamlit native widgets
        st.info("Using grid layout (full drag-and-drop coming soon)")
        
        # Display blocks in a grid
        if st.session_state.workflow["blocks"]:
            cols_per_row = 3
            for i in range(0, len(st.session_state.workflow["blocks"]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, block in enumerate(st.session_state.workflow["blocks"][i:i+cols_per_row]):
                    with cols[j]:
                        is_composer = block["type"] == "composer"
                        block_color = "rgba(232,121,42,0.2)" if is_composer else "rgba(58,32,80,0.4)"
                        block_border = "#e8792a" if is_composer else "rgba(168,85,247,0.3)"
                        
                        with st.expander(f"{'🎼 Composer' if is_composer else block['type']}", expanded=False):
                            if is_composer:
                                method = st.selectbox("Method", ["majority", "weighted", "unanimous"], 
                                                     key=f"composer_method_{block['id']}")
                                st.session_state.workflow["blocks"][i+j]["params"]["method"] = method
                            else:
                                strategy = next((s for s in STRATEGY_CATALOG if s["id"] == block["type"]), None)
                                if strategy and strategy.get("params"):
                                    for param_id, param_def in strategy["params"].items():
                                        if param_def["type"] == "int":
                                            val = st.number_input(param_def["label"], 
                                                                 min_value=param_def.get("min", 1),
                                                                 max_value=param_def.get("max", 1000),
                                                                 value=block["params"].get(param_id, param_def["default"]),
                                                                 key=f"param_{block['id']}_{param_id}")
                                        elif param_def["type"] == "float":
                                            val = st.number_input(param_def["label"],
                                                                 min_value=float(param_def.get("min", 0.0)),
                                                                 max_value=float(param_def.get("max", 1000.0)),
                                                                 value=float(block["params"].get(param_id, param_def["default"])),
                                                                 step=0.1,
                                                                 key=f"param_{block['id']}_{param_id}")
                                        elif param_def["type"] == "bool":
                                            val = st.checkbox(param_def["label"],
                                                            value=block["params"].get(param_id, param_def["default"]),
                                                            key=f"param_{block['id']}_{param_id}")
                                        else:
                                            val = st.text_input(param_def["label"],
                                                              value=str(block["params"].get(param_id, param_def["default"])),
                                                              key=f"param_{block['id']}_{param_id}")
                                        st.session_state.workflow["blocks"][i+j]["params"][param_id] = val
                            
                            # Connection management
                            if len(st.session_state.workflow["blocks"]) > 1:
                                st.caption("Connect to:")
                                other_blocks = [b for b in st.session_state.workflow["blocks"] if b["id"] != block["id"]]
                                target_options = ["None"] + [f"{b['type']} ({b['id'][:8]})" for b in other_blocks]
                                current_conn = next((c["to_id"] for c in st.session_state.workflow["connections"] if c["from_id"] == block["id"]), None)
                                current_idx = 0
                                if current_conn:
                                    for idx, b in enumerate(other_blocks):
                                        if b["id"] == current_conn:
                                            current_idx = idx + 1
                                            break
                                
                                selected = st.selectbox("→", target_options, index=current_idx, key=f"connect_{block['id']}")
                                if selected != "None":
                                    target_id = other_blocks[target_options.index(selected) - 1]["id"]
                                    # Update connections
                                    st.session_state.workflow["connections"] = [c for c in st.session_state.workflow["connections"] if c["from_id"] != block["id"]]
                                    st.session_state.workflow["connections"].append({"from_id": block["id"], "to_id": target_id})
                                elif current_conn:
                                    st.session_state.workflow["connections"] = [c for c in st.session_state.workflow["connections"] if c["from_id"] != block["id"]]
                            
                            if st.button("🗑️ Remove", key=f"remove_{block['id']}"):
                                st.session_state.workflow["blocks"] = [b for b in st.session_state.workflow["blocks"] if b["id"] != block["id"]]
                                st.session_state.workflow["connections"] = [c for c in st.session_state.workflow["connections"] 
                                                                           if c["from_id"] != block["id"] and c["to_id"] != block["id"]]
                                st.rerun()
        else:
            st.markdown("""
            <div style="width: 100%; height: 400px; background: rgba(26,15,36,0.8); border: 2px dashed rgba(168,85,247,0.3); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #b8a9d4;">
                <div style="text-align: center;">
                    <h3>Click strategies from the right sidebar to add blocks</h3>
                    <p>Configure parameters by expanding each block</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Backtest configuration panel
with st.expander("⚙️ Backtest Configuration", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value=st.session_state.backtest_config["ticker"], key="bt_ticker")
        backtest_type = st.radio("Backtest Type", ["Date Range", "# Bars"], key="bt_type", horizontal=True)
    with col2:
        if backtest_type == "Date Range":
            start_date = st.date_input("Start Date", value=pd.to_datetime(st.session_state.backtest_config["start_date"]).date(), key="bt_start")
            end_date = st.date_input("End Date", value=pd.to_datetime(st.session_state.backtest_config["end_date"]).date(), key="bt_end")
        else:
            bars_count = st.number_input("# Bars", min_value=100, max_value=100000, value=st.session_state.backtest_config["bars_count"], key="bt_bars")
    with col3:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0, key="bt_timeframe")
    
    if st.button("▶ Run Backtest", type="primary", use_container_width=True):
        # Update config
        st.session_state.backtest_config = {
            "ticker": ticker,
            "backtest_type": "date_range" if backtest_type == "Date Range" else "bars",
            "start_date": str(start_date) if backtest_type == "Date Range" else None,
            "end_date": str(end_date) if backtest_type == "Date Range" else None,
            "bars_count": bars_count if backtest_type == "# Bars" else None,
            "timeframe": timeframe,
        }
        
        # Execute backtest based on workflow
        if not st.session_state.workflow["blocks"]:
            st.warning("Add at least one strategy block to the workspace first.")
        else:
            with st.spinner("Running backtest..."):
                try:
                    from phinence.gui.runner import run_backtest_for_strategy, run_combined_backtest
                    
                    # Determine execution order from connections
                    blocks = st.session_state.workflow["blocks"]
                    connections = st.session_state.workflow["connections"]
                    
                    # Find composer blocks
                    composer_blocks = [b for b in blocks if b["type"] == "composer"]
                    strategy_blocks = [b for b in blocks if b["type"] != "composer"]
                    
                    # If composer exists, find strategies connected to it
                    if composer_blocks:
                        composer = composer_blocks[0]
                        connected_strategies = []
                        for conn in connections:
                            if conn["to_id"] == composer["id"]:
                                source_block = next((b for b in blocks if b["id"] == conn["from_id"]), None)
                                if source_block and source_block["type"] != "composer":
                                    connected_strategies.append({
                                        "id": source_block["type"],
                                        "params": source_block.get("params", {})
                                    })
                        
                        if connected_strategies:
                            # Run combined backtest with composer method
                            strategy_ids = [s["id"] for s in connected_strategies]
                            strategy_params_map = {s["id"]: s["params"] for s in connected_strategies}
                            compounding_params = composer.get("params", {})
                            
                            result = run_combined_backtest(
                                strategy_ids,
                                ticker,
                                str(start_date) if backtest_type == "Date Range" else None,
                                str(end_date) if backtest_type == "Date Range" else None,
                                data_root=REPO_ROOT / "data" / "bars",
                                strategy_params_map=strategy_params_map,
                                compounding_params=compounding_params,
                                voting_mode=compounding_params.get("method", "majority"),
                            )
                            st.session_state.backtest_results = [result]
                        else:
                            st.warning("Connect strategies to the Composer block.")
                    elif len(strategy_blocks) == 1:
                        # Single strategy
                        block = strategy_blocks[0]
                        result = run_backtest_for_strategy(
                            block["type"],
                            ticker,
                            str(start_date) if backtest_type == "Date Range" else None,
                            str(end_date) if backtest_type == "Date Range" else None,
                            data_root=REPO_ROOT / "data" / "bars",
                            strategy_params=block.get("params", {}),
                        )
                        st.session_state.backtest_results = [result]
                    elif len(strategy_blocks) > 1:
                        # Multiple strategies without composer - run individually
                        results = []
                        for block in strategy_blocks:
                            result = run_backtest_for_strategy(
                                block["type"],
                                ticker,
                                str(start_date) if backtest_type == "Date Range" else None,
                                str(end_date) if backtest_type == "Date Range" else None,
                                data_root=REPO_ROOT / "data" / "bars",
                                strategy_params=block.get("params", {}),
                            )
                            results.append(result)
                        st.session_state.backtest_results = results
                    else:
                        st.error("No strategies in workflow.")
                    
                    st.success("Backtest completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Backtest error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Results display
if st.session_state.backtest_results:
    st.subheader("Results")
    df = pd.DataFrame(st.session_state.backtest_results)
    st.dataframe(df, use_container_width=True, hide_index=True)
