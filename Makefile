# Phi-nance — Makefile
# =============================================================================
# Convenience targets for development, testing, and deployment.
#
# Usage:
#   make help        — show this help
#   make install     — install dev dependencies
#   make test        — run the full test suite
#   make lint        — run ruff linter
#   make format      — auto-format with ruff
#   make ui          — launch the Streamlit workbench
#   make backtest    — run a quick SPY backtest from CLI
#   make clean       — remove __pycache__ and .pytest_cache

.PHONY: help install install-all test test-fast test-cov lint format \
        ui backtest fetch-data runs clean dist

PYTHON  ?= python3
PIP     ?= pip
PYTEST  ?= python -m pytest
RUFF    ?= ruff

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Phi-nance Development Commands"
	@echo "  ───────────────────────────────────────────────────────────────"
	@echo "  make install       Install core + dev dependencies"
	@echo "  make install-all   Install all optional extras (ml, options)"
	@echo ""
	@echo "  make test          Run full test suite (pytest)"
	@echo "  make test-fast     Run tests, exit on first failure"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "  make lint          Lint with ruff"
	@echo "  make format        Auto-format code with ruff"
	@echo ""
	@echo "  make ui            Launch Streamlit workbench (port 8501)"
	@echo "  make backtest      Run a quick SPY CLI backtest"
	@echo "  make fetch-data    Fetch + cache SPY 1D data"
	@echo "  make runs          List recent saved runs"
	@echo ""
	@echo "  make clean         Remove build artefacts"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────────────
install:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[dev,ml,options]"

requirements:
	$(PIP) install -r requirements.txt

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -x

test-cov:
	$(PYTEST) tests/ --cov=phinance --cov-report=term-missing --cov-report=html:htmlcov

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

# ── Code quality ─────────────────────────────────────────────────────────────
lint:
	$(RUFF) check phinance/ tests/ scripts/ frontend/

format:
	$(RUFF) check --fix phinance/ tests/ scripts/ frontend/
	$(RUFF) format phinance/ tests/ scripts/ frontend/

typecheck:
	$(PYTHON) -m mypy phinance/ --ignore-missing-imports

# ── UI ────────────────────────────────────────────────────────────────────────
ui:
	streamlit run frontend/streamlit/live_workbench.py

ui-dev:
	PHINANCE_LOG_LEVEL=DEBUG streamlit run frontend/streamlit/live_workbench.py

# ── CLI workflows ─────────────────────────────────────────────────────────────
backtest:
	$(PYTHON) scripts/run_backtest.py \
		--symbol SPY \
		--start  2022-01-01 \
		--end    2024-12-31 \
		--tf     1D \
		--vendor yfinance \
		--indicators RSI MACD Bollinger \
		--blend  weighted_sum \
		--capital 100000

backtest-phiai:
	$(PYTHON) scripts/run_backtest.py \
		--symbol SPY \
		--start  2022-01-01 \
		--end    2024-12-31 \
		--tf     1D \
		--vendor yfinance \
		--indicators RSI MACD Bollinger "Dual SMA" \
		--blend  regime_weighted \
		--capital 100000 \
		--phiai

fetch-data:
	$(PYTHON) scripts/fetch_data.py \
		--symbol SPY QQQ AAPL \
		--start  2022-01-01 \
		--end    2024-12-31 \
		--tf     1D \
		--vendor yfinance

runs:
	$(PYTHON) scripts/list_runs.py --limit 20

# ── Build / dist ──────────────────────────────────────────────────────────────
dist:
	$(PYTHON) -m build

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"     -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"        -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"          -delete 2>/dev/null || true
	find . -type f -name ".coverage"      -delete 2>/dev/null || true
	@echo "Clean complete."
