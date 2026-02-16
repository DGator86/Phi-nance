# Using Agent World Model (AWM) with Phi-nance

[Agent World Model (AWM)](https://github.com/Snowflake-Labs/agent-world-model) ([DGator86 fork](https://github.com/DGator86/agent-world-model)) is a pipeline that generates **1,000+ synthetic, executable environments** for agentic reinforcement learning. Each environment is SQL-database-backed and exposed via a unified **MCP (Model Context Protocol)** interface so agents can call tools, read state, and get reward signals from verification code.

## What AWM does

1. **Scenario** → high-level description (e.g. “online shopping platform”).
2. **Tasks** → user tasks as functional requirements.
3. **Database** → SQLite schema + sample data as state backend.
4. **Interface** → Python (FastAPI) + **MCP** as action/observation space.
5. **Verification** → code that checks DB state for reward signals.

So AWM is about **synthetic tool-use environments** for training multi-turn agents, not about market data per se. The link to Phi-nance is: **use Phi-nance as one of the tools** an agent can call (e.g. “get projection”, “run backtest”) by exposing Phi-nance as an **MCP server**. Then AWM’s agent (or any MCP client) can talk to our server.

---

## How you can use AWM with Phi-nance

### Option 1: Run AWM’s agent against Phi-nance as the “environment” (recommended)

Expose Phi-nance as an MCP server with tools such as:

- `get_projection(ticker, as_of_date)` → return a `ProjectionPacket` (direction, drift, cones).
- `list_tickers()` → list symbols in the bar store.
- `run_backtest(tickers, start, end)` → run WF backtest and return summary (e.g. AUC, cone coverage).

Then run AWM’s agent demo with `--mcp_url` pointing at your Phi-nance MCP server. The agent’s “task” becomes something like: “Get the projection for SPY and summarize it” or “Run a backtest for 2024 and report the result.”

**Steps:**

1. **Install AWM** (from the [repo](https://github.com/DGator86/agent-world-model)):

   ```bash
   git clone https://github.com/DGator86/agent-world-model
   cd agent-world-model
   uv sync
   ```

2. **Start Phi-nance MCP server** (from this repo):

   ```bash
   pip install phi-nance[mcp]   # optional extra for MCP
   python -m scripts.run_mcp_server --port 8001
   ```

   Server will be at `http://localhost:8001/mcp` (or the path your script advertises).

3. **Run AWM’s agent** with your MCP URL:

   ```bash
   awm agent --task "Get the projection for SPY for today and summarize direction and confidence" \
     --mcp_url http://localhost:8001/mcp \
     --vllm_url http://localhost:8000/v1 \
     --model Snowflake/Arctic-AWM-4B
   ```

You can define tasks that require the agent to call `get_projection`, `run_backtest`, etc., and optionally add verification (e.g. “did the agent return a valid packet?”).

---

### Option 2: Use AWM’s 1K environments only (no Phi-nance in the loop)

Use AWM purely for **general** agentic RL: download the pre-built [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) environments, start any scenario’s MCP server, and run the agent against that environment. Phi-nance is not involved; this is “use AWM as intended” for non-finance tasks.

```bash
hf download Snowflake/AgentWorldModel-1K --repo-type dataset --local-dir ./outputs/
awm env start --scenario e_commerce_33 --envs_load_path outputs/gen_envs.jsonl --port 8001
awm agent --task "show me the top 10 most expensive products" --mcp_url http://localhost:8001/mcp ...
```

---

### Option 3: Use AWM’s synthesis pipeline for finance scenarios (advanced)

AWM’s pipeline (scenario → tasks → DB → API spec → MCP env → verifier) could be adapted to **finance** scenarios, e.g.:

- Scenario: “equity options research platform with bar data and projections.”
- Tasks: “Get 1d projection for SPY,” “Run a backtest for QQQ over 2024.”
- “Database” could be minimal (e.g. list of tickers, last run dates) and the real state could be Phi-nance’s bar store and projection outputs.
- Tools would wrap Phi-nance (get_projection, run_backtest, etc.) and optionally read/write a small SQLite state for the agent.

That would be a **custom finance-oriented fork** of AWM’s synthesis steps rather than using the stock AWM repo as-is.

---

## Summary

| Goal | What to do |
|------|------------|
| **Have an agent use Phi-nance (projections, backtest)** | Expose Phi-nance as an MCP server (Option 1), then run AWM’s agent with `--mcp_url` pointing at it. |
| **Train agents on generic tool-use environments** | Use AWM’s 1K environments and agent demo without Phi-nance (Option 2). |
| **Generate synthetic finance environments** | Adapt AWM’s synthesis pipeline to finance + Phi-nance tools (Option 3). |

For **Phi-nance + AWM** in practice, Option 1 is the main path: run the optional `scripts/run_mcp_server.py` (see [MCP server](#optional-phi-nance-mcp-server) below), then point AWM’s agent at `http://localhost:8001/mcp` and give it tasks that require calling our tools.

---

## Optional: Phi-nance MCP server

If the `phi-nance[mcp]` extra and `scripts/run_mcp_server.py` are present, you can start an MCP server that exposes Phi-nance tools so any MCP client (including AWM’s agent) can call them:

```bash
pip install phi-nance[mcp]
python -m scripts.run_mcp_server --port 8001
# MCP endpoint: http://localhost:8001/mcp (or as printed)
```

Tools exposed:

- **list_tickers** — no args → list of tickers in the bar store.
- **get_projection** — `(ticker, as_of_date)` → human-readable summary (direction, drift, cone); use YYYY-MM-DD or omit for today.
- **run_backtest** — `(tickers, start, end)` → summary (mean OOS AUC, 75% cone, gate). `tickers` comma-separated, e.g. "SPY, QQQ".

The agent can then be given tasks like “Get today’s projection for SPY and tell me the daily direction” or “Run a backtest for SPY and QQQ for 2024 and report the gate result.”

---

## References

- AWM paper: [arXiv:2602.10090](https://arxiv.org/abs/2602.10090)
- Code: [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model), [DGator86/agent-world-model](https://github.com/DGator86/agent-world-model)
- AgentWorldModel-1K: [Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K)
- Arctic-AWM models: [Snowflake/Arctic-AWM-4B](https://huggingface.co/Snowflake/Arctic-AWM-4B), etc.
