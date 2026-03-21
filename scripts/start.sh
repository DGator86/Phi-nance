#!/bin/bash
# Phi-nance: pull latest and start
cd "$(dirname "$0")"
git pull origin MAIN 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true
python -m streamlit run app_streamlit/live_workbench.py --server.port 8501 --server.address 0.0.0.0
