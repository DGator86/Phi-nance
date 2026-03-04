# Phi-nance Strategy Lab — GUI for Everyone

The **Strategy Lab** is a simple web interface so you can compare trading strategies and get help from an AI agent **without writing any code**.

**If it's not working:** see **[RUN_STRATEGY_LAB.md](../RUN_STRATEGY_LAB.md)** in the project root for step-by-step setup and troubleshooting.

## How to open the GUI

1. **Install the GUI** (one time). In a terminal opened in the project folder:
   ```bash
   pip install phi-nance[gui]
   ```
   (If `pip` isn't recognized, try: `python -m pip install "phi-nance[gui]"`.)

2. **Start the app**:
   ```bash
   streamlit run scripts/app.py
   ```
   (Or: `python -m streamlit run scripts/app.py`.)

   Your browser should open to **Phi-nance Strategy Lab**. If not, look for `Local URL: http://localhost:8501` in the terminal and open that link.

**Windows:** You can double-click `run_strategy_lab.bat` in the project folder to install (if needed) and start the app.

## What you can do

### 1. Pick strategies

In the **left sidebar** you’ll see a list of strategies with checkboxes:

- **Buy & Hold** — Buy once and hold. Good baseline to compare others to.
- **SMA Crossover** — Buys when a short moving average crosses above a long one; sells when it crosses below.
- **Phi-nance Projection** — Uses our market projection (liquidity, regime, sentiment) to decide daily direction.

**Check one or more** strategies. You’ll compare only the ones you select.

### 2. Set symbol and dates

Still in the sidebar:

- **Symbol** — e.g. `SPY`, `QQQ`, `AAPL`. This is the ticker you’re testing.
- **Start date** and **End date** — e.g. `2024-01-01` and `2024-06-30`. Use this range for the backtest.

### 3. Run backtest

Click the big **Run backtest** button in the sidebar. The app will run each selected strategy over the dates you chose and show a **Results** table with:

- **Return %** — How much the strategy gained or lost (as a percentage).
- **Sharpe Ratio** — Risk‑adjusted performance (higher is better, in simple terms).
- **Max. Drawdown %** — Largest peak‑to‑trough drop (lower absolute value is less painful).
- **# Trades** — Number of trades.
- **Win Rate %** — Percentage of trades that made money.

Below the table you’ll see **Best overall in this test** — the strategy with the highest Sharpe Ratio in this run.

### 4. Ask the Agent (learning)

Scroll down to **Ask the Agent**. Type a question in plain English, for example:

- “Which strategy should I use?”
- “What does Sharpe Ratio mean?”
- “Why did Buy & Hold do better than the others?”

The agent can see your **last backtest results**, so it can give answers that refer to the strategies you just ran. No programming needed.

**To enable the agent:** put your OpenAI API key in a file named `.env` in the project root, with a line like:

```text
OPENAI_API_KEY=your-key-here
```

If you don’t set a key, the app will tell you how to add one. You can still use the rest of the GUI (strategies and backtest) without it.

## Tips for non-programmers

- You don’t need to edit any code. Use only the web page and the sidebar.
- Start with **one symbol** (e.g. SPY) and **two strategies** (e.g. Buy & Hold and SMA Crossover). Click **Run backtest** and read the table.
- Use **Ask the Agent** to explain any column or to get a simple suggestion (e.g. “try a longer date range” or “try QQQ”).
- The app uses **synthetic data** if you don’t have real data in `data/bars`. So it works out of the box for learning; results are illustrative.

## Optional: Agent World Model (AWM)

For advanced “learning” where an agent can run tools (e.g. run a backtest or get a projection) via MCP, you can run our **MCP server** and connect it to [Agent World Model](https://github.com/DGator86/agent-world-model). See [AGENT_WORLD_MODEL.md](AGENT_WORLD_MODEL.md) for that setup. The Strategy Lab’s **Ask the Agent** is simpler: it only needs an OpenAI API key and uses your last backtest results as context.
