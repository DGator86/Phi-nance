#!/bin/bash
# Phi-nance: pull latest and start Live Backtest Workbench
cd "$(dirname "$0")"
git pull origin MAIN
source venv/bin/activate 2>/dev/null || true
python -m streamlit run app_streamlit/premium_dashboard.py --server.port 8501 --server.address 0.0.0.0
