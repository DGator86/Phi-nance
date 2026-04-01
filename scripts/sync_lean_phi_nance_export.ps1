<#
.SYNOPSIS
  Copy Phi-nance QuantConnect bundle into a Lean CLI project (data/ohlcv.csv + signal_card.json).

.NOTES
  This file lives under the Phi-nance repo (scripts/). It is not in LeanWorkspace.
  Call it with a full path, or cd to the Phi-nance repo root first.

.EXAMPLE
  # From Phi-nance repo root:
  .\scripts\sync_lean_phi_nance_export.ps1 `
    -ExportDir ".\exports\qc_SPY_daily_uw" `
    -LeanProject "C:\Users\you\LeanWorkspace\PhiNanceExported"

.EXAMPLE
  # From LeanWorkspace (use full path to the script):
  & "C:\Users\you\Phi-nance\scripts\sync_lean_phi_nance_export.ps1" `
    -ExportDir "C:\Users\you\Phi-nance\exports\qc_SPY_daily_uw" `
    -LeanProject "C:\Users\you\LeanWorkspace\PhiNanceExported"
#>
param(
    [Parameter(Mandatory = $true)]
    [string] $ExportDir,
    [Parameter(Mandatory = $true)]
    [string] $LeanProject
)

$ErrorActionPreference = "Stop"
$csv = Join-Path $ExportDir "ohlcv.csv"
if (-not (Test-Path $csv)) {
    throw "Missing ohlcv.csv under: $ExportDir"
}

$dataDir = Join-Path $LeanProject "data"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
Copy-Item $csv (Join-Path $dataDir "ohlcv.csv") -Force

$card = Join-Path $ExportDir "signal_card.json"
if (Test-Path $card) {
    Copy-Item $card (Join-Path $LeanProject "signal_card.json") -Force
}

Write-Host "Synced ohlcv.csv -> $dataDir\ohlcv.csv"
if (Test-Path $card) {
    Write-Host "Synced signal_card.json -> $LeanProject\signal_card.json"
}
Write-Host "Run from LeanWorkspace: lean backtest PhiNanceExported   (omit --python-venv for Docker on Windows)"
