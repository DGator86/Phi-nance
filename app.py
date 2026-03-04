#!/usr/bin/env python3
"""
Phi-nance Strategy Lab — standalone GUI.

Run: python3 -m streamlit run app.py
"""

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np

# Strategy choices
STRATEGY_CHOICES = [
    {"id": "buy_and_hold", "name": "Buy & Hold", "description": "Buy once and hold. Simple baseline."},
    {"id": "sma_cross", "name": "SMA Crossover", "description": "Buy when short average crosses above long; sell when it crosses below."},
]

# Voting modes for combining strategies
VOTING_MODES = [
    {"id": "majority", "name": "Majority Vote", "description": "Buy/sell when most strategies agree"},
    {"id": "unanimous", "name": "Unanimous", "description": "Buy/sell only when all strategies agree"},
    {"id": "weighted", "name": "Weighted Average", "description": "Weight signals by strategy performance"},
]

def make_synthetic_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    dr = pd.date_range(start=start, end=end, freq="B")
    price = 100.0
    rows = []
    np.random.seed(hash(ticker) % 10000)
    for d in dr:
        ret = np.random.randn() * 0.01
        open_p = price
        price = price * (1 + ret)
        high = max(open_p, price) * 1.01
        low = min(open_p, price) * 0.99
        vol = int(np.random.lognormal(0, 2) * 1000)
        rows.append({"Open": open_p, "High": high, "Low": low, "Close": price, "Volume": vol})
    df = pd.DataFrame(rows, index=dr)
    df.index.name = "datetime"
    return df

def _get_strategy_signal(strategy_id: str, bt_df: pd.DataFrame, idx: int) -> int:
    """Get signal from a strategy at a given index. Returns: 1 = buy, -1 = sell, 0 = hold/neutral"""
    if strategy_id == "buy_and_hold":
        return 1 if idx == 0 else 0
    
    elif strategy_id == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
            def SMA(values, n):
                return pd.Series(values).rolling(n).mean()
        
        if idx < 20:
            return 0
        
        closes = bt_df["Close"].iloc[:idx+1]
        ma1 = SMA(closes, 10)
        ma2 = SMA(closes, 20)
        
        if len(ma1) < 2 or len(ma2) < 2:
            return 0
        
        if ma1.iloc[-2] <= ma2.iloc[-2] and ma1.iloc[-1] > ma2.iloc[-1]:
            return 1  # Golden cross
        elif ma1.iloc[-2] >= ma2.iloc[-2] and ma1.iloc[-1] < ma2.iloc[-1]:
            return -1  # Death cross
        return 0
    
    return 0


def run_backtest_for_strategy(strategy_id: str, ticker: str, start: str, end: str) -> dict[str, Any]:
    """Run one backtest and return metrics."""
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        return {"Strategy": strategy_id, "Error": "Install: pip install backtesting"}
    
    bt_df = make_synthetic_data(ticker, start, end)
    if bt_df.empty or len(bt_df) < 20:
        name = next((c["name"] for c in STRATEGY_CHOICES if c["id"] == strategy_id), strategy_id)
        return {"Strategy": name, "Error": "Not enough data"}
    
    if strategy_id == "buy_and_hold":
        class BuyAndHold(Strategy):
            def init(self):
                self._bought = False
            def next(self):
                if not self._bought:
                    self.buy()
                    self._bought = True
        StrategyClass = BuyAndHold
    elif strategy_id == "sma_cross":
        try:
            from backtesting.test import SMA
        except ImportError:
            def SMA(values, n):
                return pd.Series(values).rolling(n).mean()
        class SmaCross(Strategy):
            n1 = 10
            n2 = 20
            def init(self):
                self.ma1 = self.I(SMA, self.data.Close, self.n1)
                self.ma2 = self.I(SMA, self.data.Close, self.n2)
            def next(self):
                if len(self.data) < self.n2:
                    return
                if crossover(self.ma1, self.ma2):
                    self.buy()
                elif crossover(self.ma2, self.ma1):
                    self.position.close()
        StrategyClass = SmaCross
    else:
        return {"Strategy": strategy_id, "Error": "Unknown strategy"}
    
    bt = Backtest(bt_df, StrategyClass, commission=0.002, exclusive_orders=True, trade_on_close=True, finalize_trades=True)
    stats = bt.run()
    name = next((c["name"] for c in STRATEGY_CHOICES if c["id"] == strategy_id), strategy_id)
    if not isinstance(stats, pd.Series):
        return {"Strategy": name, "Return [%]": 0, "Sharpe Ratio": 0, "Max. Drawdown [%]": 0}
    
    s = stats
    def _num(x, default=0):
        v = s.get(x, default)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    
    return {
        "Strategy": name,
        "Return [%]": round(_num("Return [%]"), 2),
        "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
        "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
        "# Trades": int(s.get("# Trades", 0) or 0),
        "Win Rate [%]": round(_num("Win Rate [%]"), 1),
    }

st.set_page_config(
    page_title="Phi-nance Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

st.sidebar.title("📈 Strategy Lab")
st.sidebar.markdown("Pick strategies and dates, then run a backtest to compare them.")

st.sidebar.subheader("1. Pick your strategies")
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

st.title("Phi-nance Strategy Lab")
st.markdown("Compare strategies side by side and see which one fits best. No code required.")

if run_button:
    if not selected:
        st.warning("Please select at least one strategy in the sidebar.")
    else:
        results = []
        progress = st.progress(0, text="Running backtests...")
        
        if run_mode == "Combine strategies" and len(selected) > 1:
            # Run combined strategy
            progress.progress(0.5, text=f"Running combined strategy ({len(selected)} strategies)...")
            
            bt_df = make_synthetic_data(ticker, start, end)
            if bt_df.empty or len(bt_df) < 20:
                results.append({"Strategy": "+".join(selected), "Error": "Not enough data"})
            else:
                # Pre-compute signals
                signals_by_strategy = {}
                for sid in selected:
                    signals = []
                    for idx in range(len(bt_df)):
                        signal = _get_strategy_signal(sid, bt_df, idx)
                        signals.append(signal)
                    signals_by_strategy[sid] = signals
                
                # Create combined strategy
                from backtesting import Backtest, Strategy
                
                class CombinedStrategy(Strategy):
                    def init(self):
                        pass
                    
                    def next(self):
                        idx = len(self.data) - 1
                        if idx < 0:
                            return
                        
                        signals = []
                        for sid in selected:
                            if sid in signals_by_strategy and idx < len(signals_by_strategy[sid]):
                                signals.append(signals_by_strategy[sid][idx])
                            else:
                                signals.append(0)
                        
                        if voting_mode == "majority":
                            buy_votes = sum(1 for s in signals if s > 0)
                            sell_votes = sum(1 for s in signals if s < 0)
                            if buy_votes > sell_votes and buy_votes > len(signals) / 2:
                                if not self.position:
                                    self.buy()
                            elif sell_votes > buy_votes and sell_votes > len(signals) / 2:
                                if self.position:
                                    self.position.close()
                        elif voting_mode == "unanimous":
                            all_buy = all(s > 0 for s in signals)
                            all_sell = all(s < 0 for s in signals)
                            if all_buy and not self.position:
                                self.buy()
                            elif all_sell and self.position:
                                self.position.close()
                        elif voting_mode == "weighted":
                            weighted_signal = sum(signals) / len(signals) if signals else 0
                            if weighted_signal > 0.3:
                                if not self.position:
                                    self.buy()
                            elif weighted_signal < -0.3:
                                if self.position:
                                    self.position.close()
                
                bt = Backtest(bt_df, CombinedStrategy, commission=0.002, exclusive_orders=True, trade_on_close=True, finalize_trades=True)
                stats = bt.run()
                
                strategy_names = [next((c["name"] for c in STRATEGY_CHOICES if c["id"] == sid), sid) for sid in selected]
                combined_name = " + ".join(strategy_names)
                
                if not isinstance(stats, pd.Series):
                    results.append({"Strategy": combined_name, "Return [%]": 0, "Sharpe Ratio": 0, "Max. Drawdown [%]": 0})
                else:
                    s = stats
                    def _num(x, default=0):
                        v = s.get(x, default)
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return default
                        return float(v)
                    
                    results.append({
                        "Strategy": combined_name,
                        "Return [%]": round(_num("Return [%]"), 2),
                        "Sharpe Ratio": round(_num("Sharpe Ratio"), 2),
                        "Max. Drawdown [%]": round(_num("Max. Drawdown [%]"), 2),
                        "# Trades": int(s.get("# Trades", 0) or 0),
                        "Win Rate [%]": round(_num("Win Rate [%]"), 1),
                    })
        else:
            # Run each strategy individually
            for i, strategy_id in enumerate(selected):
                progress.progress((i + 1) / len(selected), text=f"Running {strategy_id}...")
                r = run_backtest_for_strategy(strategy_id, ticker, start, end)
                results.append(r)
        
        progress.empty()
        st.session_state.backtest_results = results
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

if st.session_state.backtest_results:
    st.subheader("📊 Results")
    df = pd.DataFrame(st.session_state.backtest_results)
    if "Error" in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
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
            "To use the learning agent, add your OpenAI API key to a `.env` file: "
            "`OPENAI_API_KEY=your-key`. Then restart the app. You can still run backtests above."
        )
    st.session_state.agent_messages.append({"role": "assistant", "content": reply})

for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
