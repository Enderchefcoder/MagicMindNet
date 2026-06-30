# One-command dev environment bootstrap (Windows).
# Creates .venv, installs dev deps, builds the extension, runs the merge gate.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.12+ required on PATH."
}

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -U pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m maturin develop --release -m crates/mmn-py/Cargo.toml

Write-Host "Running verify_gate..."
& .\scripts\verify_gate.ps1
