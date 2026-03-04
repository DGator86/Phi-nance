#!/usr/bin/env python3
"""
Phi-nance Strategy Lab — simple GUI for everyone.

  streamlit run scripts/app.py

No programming needed: pick strategies, run backtest, see which performs best, and ask the Agent for help.
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

from phinence.gui.runner import run_backtest_for_strategy, run_combined_backtest, STRATEGY_CHOICES

st.set_page_config(
    page_title="Phi-nance Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state for results and chat
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# ---- Sidebar: simple settings ----
st.sidebar.title("📈 Strategy Lab")
st.sidebar.markdown("Pick strategies and dates, then run a backtest to compare them.")

st.sidebar.subheader("1. Pick your strategies")
st.sidebar.caption("Select one or more to compare.")
selected = []
for choice in STRATEGY_CHOICES:
    if st.sidebar.checkbox(choice["name"], key=choice["id"], help=choice["description"]):
        selected.append(choice["id"])

if not selected:
    st.sidebar.info("👆 Check at least one strategy above.")

st.sidebar.subheader("2. Run mode")
run_mode = st.sidebar.radio(
    "How to run strategies:",
    ["Compare individually", "Combine strategies"],
    help="Compare: Run each separately and compare results. Combine: Run as one combined strategy using voting."
)

voting_mode = "majority"
if run_mode == "Combine strategies" and len(selected) > 1:
    from phinence.gui.runner import VOTING_MODES
    voting_mode = st.sidebar.selectbox(
        "Voting mode:",
        options=[vm["id"] for vm in VOTING_MODES],
        format_func=lambda x: next((vm["name"] for vm in VOTING_MODES if vm["id"] == x), x),
        help="How to combine signals from multiple strategies"
    )

st.sidebar.subheader("3. Symbol & dates")
ticker = st.sidebar.text_input("Symbol", value="SPY", help="e.g. SPY, QQQ, AAPL").strip() or "SPY"
start = st.sidebar.text_input("Start date", value="2024-01-01", help="YYYY-MM-DD")
end = st.sidebar.text_input("End date", value="2024-06-30", help="YYYY-MM-DD")

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run backtest", type="primary", use_container_width=True)

# ---- Main area ----
st.title("Phi-nance Strategy Lab")
st.markdown("Compare strategies side by side and see which one fits best. No code required.")

# Run backtests when user clicks
if run_button:
    if not selected:
        st.warning("Please select at least one strategy in the sidebar.")
    else:
        results = []
        progress = st.progress(0, text="Running backtests...")
        
        if run_mode == "Combine strategies" and len(selected) > 1:
            # Run combined strategy
            progress.progress(0.5, text=f"Running combined strategy ({len(selected)} strategies)...")
            r = run_combined_backtest(
                selected, ticker, start, end, 
                voting_mode=voting_mode,
                data_root=REPO_ROOT / "data" / "bars"
            )
            results.append(r)
        else:
            # Run each strategy individually
            for i, strategy_id in enumerate(selected):
                progress.progress((i + 1) / len(selected), text=f"Running {strategy_id}...")
                r = run_backtest_for_strategy(strategy_id, ticker, start, end, data_root=REPO_ROOT / "data" / "bars")
                results.append(r)
        
        progress.empty()
        st.session_state.backtest_results = results
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# Show results table
if st.session_state.backtest_results:
    st.subheader("📊 Results")
    df = pd.DataFrame(st.session_state.backtest_results)
    if "Error" in df.columns:
        # Show errors in table; hide error column for display if all have errors
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Best overall: highlight row with best Sharpe (or best Return if no Sharpe)
        if "Sharpe Ratio" in df.columns and len(df) > 0:
            best_idx = df["Sharpe Ratio"].idxmax()
            best_name = df.loc[best_idx, "Strategy"]
            st.success(f"**Best overall in this test:** **{best_name}** (highest Sharpe Ratio)")
        elif "Return [%]" in df.columns and len(df) > 0:
            best_idx = df["Return [%]"].idxmax()
            best_name = df.loc[best_idx, "Strategy"]
            st.success(f"**Highest return in this test:** **{best_name}**")
    st.session_state["last_results_summary"] = df.to_string() if isinstance(df, pd.DataFrame) else str(st.session_state.backtest_results)
else:
    st.info("👈 Select one or more strategies in the sidebar and click **Run backtest** to see results.")
    st.session_state["last_results_summary"] = ""

# ---- Ask the Agent (learning) ----
st.markdown("---")
st.subheader("🤖 Ask the Agent")
st.caption("Get simple explanations and suggestions. The agent can see your last backtest results.")
agent_input = st.chat_input("Ask anything (e.g. Which strategy should I use? What does Sharpe Ratio mean?)")

if agent_input:
    st.session_state.agent_messages.append({"role": "user", "content": agent_input})
    summary = st.session_state.get("last_results_summary", "")
    system = (
        "You are a friendly trading assistant for non-experts. Explain in plain language, no jargon. "
        "Keep answers short and helpful. If the user shared backtest results, you can refer to them to suggest which strategy might be best or what to try next."
    )
    user_context = f"User's last backtest results:\n{summary}\n\nUser question: {agent_input}" if summary else agent_input
    reply = "I'm here to help. To get answers that use your backtest results, run a backtest first, then ask your question."
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_context},
                ],
                max_tokens=400,
            )
            if r.choices:
                reply = r.choices[0].message.content or reply
        except Exception as e:
            reply = f"I couldn't reach the assistant right now ({e}). Try again or add OPENAI_API_KEY to your .env."
    else:
        reply = (
            "To use the learning agent, add your OpenAI API key to a `.env` file in the project root: "
            "`OPENAI_API_KEY=your-key`. Then restart the app. You can still run backtests and compare strategies above."
        )
    st.session_state.agent_messages.append({"role": "assistant", "content": reply})

for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
