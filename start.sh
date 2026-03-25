#!/bin/bash
# Phi-nance: activate venv and start the QuantConnect export API (headless).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "ERROR: venv not found. Run 'python3 -m venv venv && pip install -r requirements.txt' first."
    exit 1
fi

export PHINANCE_QC_EXPORT_DIR="${PHINANCE_QC_EXPORT_DIR:-./exports/qc_bundles}"
mkdir -p "$PHINANCE_QC_EXPORT_DIR"

echo "Starting QC export API on http://0.0.0.0:8080 (export dir: $PHINANCE_QC_EXPORT_DIR)"
exec uvicorn phi.api.qc_export:app --host 0.0.0.0 --port 8080
