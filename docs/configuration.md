# Configuration

Phi-nance is environment-driven. Copy `.env.example` to `.env` and set required keys.

## Core runtime variables

- `DATA_CACHE_DIR`: cache storage directory.
- `RUNS_DIR`: run artifact output directory.
- `LOGS_DIR`: log output directory.
- `LOG_LEVEL`: logger verbosity.
- `DEBUG`: enables extra debug detail in UI error rendering.
- `DATA_CACHE_ROOT`: deprecated alias for `DATA_CACHE_DIR`.

## PhiAI defaults

- `PHIAI_DEFAULT_N_TRIALS`
- `PHIAI_PARALLEL_JOBS`
- `PHIAI_WALK_FORWARD_WINDOWS`

## Data/API providers

- `AV_API_KEY`
- `MASSIVE_API_KEY`
- `TRADIER_ACCESS_TOKEN`
- `MARKETDATAAPP_API_TOKEN` (optional options-chain enrichment)
- `FINNHUB_API_KEY` (optional)
- `STOCKDATA_API_KEY` (optional)

## AI integrations

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`

## Operational / optional

- `UPDATE_CHECK_CADENCE`
- `PUBLIC_API_KEY`
- `DIGITALOCEAN_TOKEN`
- `AV_MCP_TOKEN`
- `IS_BACKTESTING` (set by app bootstrap for lumibot-safe behavior)

