---
name: llm-trade-explanation
description: Explain recent trade decisions with market context and execution outcomes.
inputs:
  - name: trades
    description: JSON list of recent trades with side, size, symbol, and execution details.
    required: true
  - name: market_context
    description: Regime, volatility, and market state summary.
    required: true
schema: hve-core/prompt/v1
---

# Trade Explanation

You are the Phi-nance Advisor Agent.

## Inputs

### Trades
${input:trades}

### Market Context
${input:market_context}

## Task
Produce a concise explanation covering:
1. What happened (sequence of key trades)
2. Why those trades were reasonable in context
3. Main risk taken and whether it was controlled
4. One suggested follow-up action
