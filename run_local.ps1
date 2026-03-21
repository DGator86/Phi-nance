# Phi-nance - Streamlit on localhost (Windows-friendly; avoids 0.0.0.0 browser issues)
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "venv not found. Create it from repo root: python -m venv venv && .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

function Test-PortInUse([int]$p) {
    try {
        $rows = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        return $null -ne $rows
    } catch {
        return $false
    }
}

function Show-PortOwner([int]$p) {
    try {
        Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            ForEach-Object {
                $proc = Get-Process -Id $_ -ErrorAction SilentlyContinue
                if ($proc) { Write-Host "  PID $($proc.Id): $($proc.ProcessName)" }
            }
    } catch {
        # ignore
    }
}

$chosen = $Port
$maxPort = $Port + 20
while (Test-PortInUse $chosen) {
    if ($chosen -eq $Port) {
        Write-Host "Port $chosen is already in use (often a leftover Streamlit)."
        Show-PortOwner $chosen
    }
    $chosen++
    if ($chosen -gt $maxPort) {
        Write-Host "No free port between $Port and $maxPort. Close the other app or pick a port: .\run_local.ps1 -Port 8600"
        exit 1
    }
}

if ($chosen -ne $Port) {
    Write-Host "Using first free port: $chosen"
}

Write-Host "Starting Streamlit - keep this window open. Open: http://localhost:$chosen"
& .\venv\Scripts\python.exe -m streamlit run app_streamlit/live_workbench.py --server.port $chosen --server.address 127.0.0.1
