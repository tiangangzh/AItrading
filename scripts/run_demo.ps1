# One-shot setup + demo for Windows PowerShell.
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1          -> offline demo (no keys needed)
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Free    -> also install free real-data extras
param([switch]$Free)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null
if ($Free) {
    python -m pip install -e ".[free]"
} else {
    python -m pip install -e .
}
aitrading demo
