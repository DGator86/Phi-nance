<#
.SYNOPSIS
  Copy Phi-nance QuantConnect bundle into a Lean CLI project (workspace data/ + signal_card.json).

.NOTES
  This file lives under the Phi-nance repo (scripts/). Running from LeanWorkspace:
  & "C:\path\to\Phi-nance\scripts\sync_lean_phi_nance_export.ps1" -PhiNanceRoot "..." -LeanProject "..."

.EXAMPLE
  # Explicit export folder:
  .\scripts\sync_lean_phi_nance_export.ps1 -ExportDir ".\exports\qc_SPY_daily_uw" -LeanProject "C:\Users\you\LeanWorkspace\PhiNanceExported"

.EXAMPLE
  # Auto-pick newest ohlcv.csv under <repo>\exports (solves “OneDrive has no export”):
  & "C:\Users\you\Phi-nance\scripts\sync_lean_phi_nance_export.ps1" `
    -PhiNanceRoot "C:\Users\you\Phi-nance" `
    -LeanProject "C:\Users\you\LeanWorkspace\PhiNanceExported"
#>
param(
    [string] $ExportDir,
    [string] $PhiNanceRoot,
    [Parameter(Mandatory = $true)]
    [string] $LeanProject
)

if (-not $ExportDir) {
    if (-not $PhiNanceRoot) {
        throw "Provide -ExportDir or -PhiNanceRoot (to search under exports\ for ohlcv.csv)."
    }
    $exportsRoot = Join-Path $PhiNanceRoot "exports"
    if (-not (Test-Path $exportsRoot)) {
        throw "No exports folder at: $exportsRoot. Run scripts/export_quantconnect_bundle.py first."
    }
    $found = Get-ChildItem -Path $exportsRoot -Recurse -Filter "ohlcv.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $found) {
        throw "No ohlcv.csv under $exportsRoot. Run: python scripts/export_quantconnect_bundle.py --symbol SPY --start 2022-01-01 --end 2024-12-31 --timeframe 1D --out-dir ./exports/qc_SPY_1D"
    }
    $ExportDir = $found.Directory.FullName
    Write-Host "Using export folder: $ExportDir"
}

$ErrorActionPreference = "Stop"
$csv = Join-Path $ExportDir "ohlcv.csv"
if (-not (Test-Path $csv)) {
    throw "Missing ohlcv.csv under: $ExportDir"
}

# Lean CLI resolves LocalFile custom data via Globals.DataFolder = workspace /data
# (lean.json "data-folder"), NOT only <project>/data/.
$workspaceRoot = Split-Path -Parent $LeanProject
$workspaceData = Join-Path $workspaceRoot "data"
New-Item -ItemType Directory -Path $workspaceData -Force | Out-Null
Copy-Item $csv (Join-Path $workspaceData "ohlcv.csv") -Force

# AddEquity("SPY") needs Lean-format equity daily zip (custom PHI CSV alone is not enough).
$equityDaily = Join-Path $workspaceData "equity\usa\daily"
New-Item -ItemType Directory -Path $equityDaily -Force | Out-Null
$converter = Join-Path $PSScriptRoot "lean_spy_daily_zip_from_ohlcv.py"
$spyZip = Join-Path $equityDaily "spy.zip"
$py = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
& $py $converter --ohlcv $csv --out-zip $spyZip
if ($LASTEXITCODE -ne 0) {
    throw "lean_spy_daily_zip_from_ohlcv.py failed (exit $LASTEXITCODE). Ensure Python 3 is on PATH."
}

# Optional mirror under project (documentation only; engine uses workspace path).
$projectData = Join-Path $LeanProject "data"
New-Item -ItemType Directory -Path $projectData -Force | Out-Null
Copy-Item $csv (Join-Path $projectData "ohlcv.csv") -Force
$projectEquity = Join-Path $projectData "equity\usa\daily"
New-Item -ItemType Directory -Path $projectEquity -Force | Out-Null
Copy-Item $spyZip (Join-Path $projectEquity "spy.zip") -Force

$card = Join-Path $ExportDir "signal_card.json"
if (Test-Path $card) {
    Copy-Item $card (Join-Path $LeanProject "signal_card.json") -Force
}

Write-Host "Synced ohlcv.csv -> $workspaceData\ohlcv.csv (workspace data for Lean CLI)"
Write-Host "Wrote Lean equity daily -> $spyZip (required for AddEquity SPY orders)"
Write-Host "Mirrored ohlcv.csv -> $projectData\ohlcv.csv"
Write-Host "Mirrored spy.zip -> $projectEquity\spy.zip"
if (Test-Path $card) {
    Write-Host "Synced signal_card.json -> $LeanProject\signal_card.json"
}
Write-Host "Run from LeanWorkspace: lean backtest PhiNanceExported   (omit --python-venv for Docker on Windows)"
