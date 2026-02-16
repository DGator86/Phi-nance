# What's Running & What You Can Do

## ✅ Strategy Lab GUI — Starting Now

The **Strategy Lab GUI** is starting. Look for output like:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Open that URL in your browser** to use the app.

---

## What You Can Do in the Strategy Lab

1. **Pick strategies** (left sidebar):
   - ✅ Buy & Hold
   - ✅ SMA Crossover  
   - ⚠️ Phi-nance Projection (needs src/phinence modules)

2. **Set symbol & dates**:
   - Symbol: `SPY` (or QQQ, AAPL, etc.)
   - Start: `2024-01-01`
   - End: `2024-06-30`

3. **Click "Run backtest"** → See results table with:
   - Return %
   - Sharpe Ratio
   - Max Drawdown %
   - # Trades
   - Win Rate %

4. **See "Best overall"** → Strategy with highest Sharpe Ratio

5. **Ask the Agent** (scroll down):
   - Type questions like "Which strategy should I use?"
   - Needs `OPENAI_API_KEY` in `.env` to work

---

## Other Things You Can Run

### Lumibot Dashboard (Prediction Accuracy)
```bash
python3 -m streamlit run dashboard.py
```
Has 10+ strategies (Momentum, RSI, Bollinger, MACD, Wyckoff, etc.) and measures **prediction accuracy** instead of P&L.

### Command Line Backtests
```bash
# Using backtesting.py
python3 scripts/run_backtesting_py.py --strategy sma_cross --ticker SPY

# Using Lumibot
python3 scripts/run_lumibot_backtest.py --strategy buy_and_hold --tickers SPY

# Phi-nance native WF backtest
python3 scripts/run_backtest.py --tickers SPY QQQ
```

### MCP Server (for Agent World Model)
```bash
python3 scripts/run_mcp_server.py --port 8001
```
Then connect AWM agent to `http://localhost:8001/mcp`

---

## If the GUI Doesn't Open

Check the terminal output for the URL (usually `http://localhost:8501`). If you see errors, share them and I'll help fix.
