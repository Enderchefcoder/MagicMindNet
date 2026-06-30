# Ruff lint — run from repo root with dev deps installed
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Get-Command ruff -ErrorAction SilentlyContinue)) {
    Write-Host "ruff not on PATH; install dev extras: pip install -e '.[dev]'"
    exit 1
}

ruff check python tests examples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "lint: OK"
