# LLM Integration (Phase 6)

Phase 6 introduces an optional LLM advisor layer for reasoning and explainability.

## Components

- `phinance/llm/client.py`: backend abstraction with Ollama, OpenAI, and dummy fallback.
- `phinance/llm/prompts.py`: prompt loading and template rendering from `.github/prompts`.
- `phinance/llm/advisor.py`: advisor agent methods for trade explanation, risk report, and strategy review.
- `phinance/llm/utils.py`: prompt formatting/truncation helpers.

## Configuration

Edit `configs/llm_config.yaml`:

```yaml
llm:
  enabled: false
  backend: ollama
  model: llama3
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 800
```

### Backends

- **Ollama** (local):
  1. Install Ollama from https://ollama.com/download
  2. `ollama pull llama3`
  3. `ollama serve`
- **OpenAI** (cloud): set `backend: openai` and provide `api_key` in config or `OPENAI_API_KEY` env var.

## Runtime behavior

- If disabled (`enabled: false`) or backend fails, advisor returns fallback text and trading continues.
- Headroom compression is attempted before requests. If unavailable, prompts are sent as-is.

## Live integration

- `LiveEngine` can instantiate advisor with `{"advisor_enabled": true}`.
- The dashboard can request an on-demand explanation and display the most recent advisor report.
