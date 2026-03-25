#!/bin/bash
# VPS-style start from repo root (same as ./start.sh at project root).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "ERROR: venv not found at $ROOT/venv"
    exit 1
fi
export PHINANCE_QC_EXPORT_DIR="${PHINANCE_QC_EXPORT_DIR:-$ROOT/exports/qc_bundles}"
mkdir -p "$PHINANCE_QC_EXPORT_DIR"
exec uvicorn phi.api.qc_export:app --host 0.0.0.0 --port 8080
