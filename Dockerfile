# Phi-nance — headless research + QuantConnect export API (no Streamlit).
#
#   docker build -t phinance-qc:latest .
#   docker run -p 8080:8080 -e PHINANCE_QC_EXPORT_DIR=/exports -v qc_exports:/exports --env-file .env phinance-qc:latest
#
# Default command: FastAPI bundle exporter. Override to run scripts, pytest, etc.

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml requirements.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Phi-nance (QuantConnect-oriented)" \
      org.opencontainers.image.description="Headless quant library + QC export API" \
      org.opencontainers.image.source="https://github.com/DGator86/Phi-nance"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd -r phinance && useradd -r -g phinance -d /app -s /sbin/nologin phinance

WORKDIR /app
COPY --chown=phinance:phinance . .
RUN pip install --no-deps -e . 2>/dev/null || true

USER phinance
ENV PHINANCE_QC_EXPORT_DIR=/exports
RUN mkdir -p /exports && chown phinance:phinance /exports

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "phi.api.qc_export:app", "--host", "0.0.0.0", "--port", "8080"]
