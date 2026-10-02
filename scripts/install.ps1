# Create a virtual environment and install AItrading with the free real-data extras (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
# (A plain `scripts\install.ps1` is refused by the default execution policy on Windows clients.)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# $ErrorActionPreference does not stop on a failing native command (py / python / pip) in Windows
# PowerShell 5.1, so every step checks $LASTEXITCODE itself.
function Invoke-Step([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Installation FAILED while trying to $What (exit code $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "Check your internet connection / proxy (HTTPS_PROXY) and Python 3.10+ (py -3 --version), then run this script again."
        exit 1
    }
}

if (-not (Test-Path ".venv")) { Invoke-Step "create the virtual environment" { py -3 -m venv .venv } }
& .\.venv\Scripts\Activate.ps1
Invoke-Step "upgrade pip" { python -m pip install --upgrade pip | Out-Null }
Invoke-Step "install AItrading" { python -m pip install -e ".[free]" }
Invoke-Step "check the installation" { aitrading --version }

Write-Host ""
Write-Host "Installed. Next:"
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  $env:SEC_USER_AGENT = "Your Name you@example.com"'
Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."      # optional: enables Claude reasoning'
Write-Host '  aitrading run "your investment observation"'
Write-Host "See docs\QUICKSTART_PC.md for details."
