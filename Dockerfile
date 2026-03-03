# ── Phi-nance Platform — Main Dockerfile ─────────────────────────────────────
# Multi-stage build:
#   builder  — installs all Python dependencies into /install
#   runtime  — slim final image that copies only the installed packages
#
# Usage (single container — Streamlit only):
#   docker build -t phinance:latest .
#   docker run -p 8501:8501 --env-file .env phinance:latest
#
# Usage (full stack via Compose):
#   docker compose up --build
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System build deps (needed for scipy, lightgbm wheel builds if no binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency files first (layer-cache friendly)
COPY pyproject.toml ./
COPY requirements.txt ./

# Install into an isolated prefix so we can copy just this tree to the runtime
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir \
        "pandas>=2.0" \
        "numpy>=1.24" \
        "pyarrow>=14.0" \
        "requests>=2.28" \
        "python-dotenv>=1.0" \
        "yfinance>=0.2.40" \
        "streamlit>=1.30.0" \
        "plotly>=5.18" \
        "pyyaml>=6.0" \
        "scipy>=1.11" \
        "scikit-learn>=1.3.0" \
        "lightgbm>=4.0.0" \
        "joblib>=1.3.0" \
        "optuna>=3.5.0" \
        "alpaca-py>=0.20.0"

# ── Stage 2: slim runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Phi-nance" \
      org.opencontainers.image.description="Open-source quant research platform" \
      org.opencontainers.image.source="https://github.com/DGator86/Phi-nance" \
      org.opencontainers.image.licenses="MIT"

# Runtime-only system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r phinance && useradd -r -g phinance -d /app -s /sbin/nologin phinance

WORKDIR /app

# Copy application source
COPY --chown=phinance:phinance . .

# Streamlit config directory
RUN mkdir -p /app/.streamlit && chown -R phinance:phinance /app/.streamlit

# Install the project itself in editable mode (no extra deps)
RUN pip install --no-deps -e . 2>/dev/null || true

USER phinance

# Health check — Streamlit responds on /healthz
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

EXPOSE 8501

# Default: run the Streamlit UI (can override with ENTRYPOINT args)
ENTRYPOINT ["streamlit", "run", "frontend/streamlit/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
