# Create a virtual environment and install AItrading with the free real-data extras (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) { py -3 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null
python -m pip install -e ".[free]"

Write-Host ""
Write-Host "Installed. Next:"
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  $env:SEC_USER_AGENT = "Your Name you@example.com"'
Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."      # optional: enables Claude reasoning'
Write-Host '  aitrading run "your investment observation"'
Write-Host "See docs\QUICKSTART_PC.md for details."
