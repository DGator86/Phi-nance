# Phi-nance - Streamlit on localhost (Windows-friendly; avoids 0.0.0.0 browser issues)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "venv not found. Create it from repo root: python -m venv venv && .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "Starting Streamlit - keep this window open. Then open: http://localhost:8501"
& .\venv\Scripts\python.exe -m streamlit run app_streamlit/live_workbench.py --server.port 8501 --server.address 127.0.0.1
