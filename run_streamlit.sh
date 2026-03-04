#!/usr/bin/env bash
# Run a Streamlit app from repo root. Usage: bash run_streamlit.sh
cd "$(dirname "$0")"
if [ -f app_v2.py ]; then python3 -m streamlit run app_v2.py "$@"; exit; fi
if [ -f app.py ]; then python3 -m streamlit run app.py "$@"; exit; fi
if [ -f dashboard.py ]; then python3 -m streamlit run dashboard.py "$@"; exit; fi
if [ -f scripts/app_v2.py ]; then python3 -m streamlit run scripts/app_v2.py "$@"; exit; fi
echo "No app found. Try: python3 -m streamlit run dashboard.py"
exit 1
