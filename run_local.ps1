# Phi-nance — headless QC export API on localhost (Windows).
param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "venv not found. From repo root: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$exportDir = if ($env:PHINANCE_QC_EXPORT_DIR) { $env:PHINANCE_QC_EXPORT_DIR } else { Join-Path $PSScriptRoot "exports\qc_bundles" }
New-Item -ItemType Directory -Force -Path $exportDir | Out-Null
$env:PHINANCE_QC_EXPORT_DIR = $exportDir

Write-Host "QC export API: http://127.0.0.1:$Port/health  (PHINANCE_QC_EXPORT_DIR=$exportDir)"
& .\venv\Scripts\python.exe -m uvicorn phi.api.qc_export:app --host 127.0.0.1 --port $Port
