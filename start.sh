#!/bin/bash
# Phi-nance: activate venv and start Streamlit
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "ERROR: venv not found. Run 'python3 -m venv venv && pip install -r requirements.txt' first."
    exit 1
fi

echo "Starting Phi-nance on port 8501..."
python -m streamlit run app_streamlit/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
