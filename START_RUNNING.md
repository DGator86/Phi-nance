# Quick Start — Run Things Now

## Option 1: Strategy Lab GUI (Recommended - easiest)

**Install dependencies:**
```bash
pip install streamlit backtesting openai
pip install pydantic pyarrow pandas numpy httpx python-dotenv
```

**Run:**
```bash
python3 -m streamlit run scripts/app.py
```

Or if `app.py` is in root:
```bash
python3 -m streamlit run app.py
```

This opens a web UI where you can:
- Pick strategies (Buy & Hold, SMA Crossover, Phi-nance Projection)
- Run backtests and compare results
- Ask the Agent questions

---

## Option 2: Lumibot Dashboard (Prediction Accuracy)

**Install:**
```bash
pip install lumibot streamlit
```

**Run:**
```bash
python3 -m streamlit run dashboard.py
```

This is the original Lumibot dashboard with 10+ strategies and prediction accuracy metrics.

---

## Option 3: Run a Backtest (Command Line)

**Using backtesting.py:**
```bash
pip install backtesting
python3 scripts/run_backtesting_py.py --strategy sma_cross --ticker SPY
```

**Using Lumibot:**
```bash
pip install lumibot
python3 scripts/run_lumibot_backtest.py --strategy buy_and_hold --tickers SPY
```

**Using Phi-nance native WF:**
```bash
python3 scripts/run_backtest.py --tickers SPY QQQ
```

---

## Option 4: MCP Server (for Agent World Model)

**Install:**
```bash
pip install mcp openai
```

**Run:**
```bash
python3 scripts/run_mcp_server.py --port 8001
```

Then connect AWM agent to `http://localhost:8001/mcp`

---

## What to Run First?

**Start with Strategy Lab GUI** — it's the simplest:
1. `pip install streamlit backtesting openai`
2. `python3 -m streamlit run scripts/app.py` (or `app.py` if in root)
3. Open browser, pick strategies, click "Run backtest"
