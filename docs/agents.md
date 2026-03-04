# Free AI Agents Setup (Ollama)

Phi-nance uses [Ollama](https://ollama.com) for free, local AI agents. No API keys needed.

## 1. Install Ollama

**Download:** [ollama.com/download](https://ollama.com/download)

- **Windows:** Run installer
- **macOS:** Run installer
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

## 2. Pull a model

Open a terminal and run:

```bash
# General purpose (fast, ~2GB)
ollama pull llama3.2

# Finance-focused (trading recommendations)
ollama pull 0xroyce/plutus

# Other options
ollama pull gemma2
ollama pull mistral
ollama pull phi3
```

## 3. Start Ollama

Ollama usually runs automatically after install. If not:

```bash
ollama serve
```

## 4. Use in Phi-nance

1. Open the **AI Agents** tab in the Live Workbench
2. Click **Check connection** — should show "Ollama is running"
3. Click **List models** to load your pulled models
4. Pick a model and type your question (regime, strategy, indicators, etc.)

## Remote / Cloud

- **Local (default):** `http://localhost:11434`
- **ollama.com cloud:** `https://ollama.com/api` — set host in the app

## Integrations

- [Ollama docs](https://docs.ollama.com)
- [Ollama API](https://docs.ollama.com/api/introduction)
- [Plutus model](https://ollama.com/0xroyce/plutus) — finance-trained
