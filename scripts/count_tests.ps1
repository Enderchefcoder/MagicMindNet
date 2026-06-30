# Print approximate Rust + Python test counts (from repo root).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$PY = & "$PSScriptRoot\venv_python.ps1"

$rustUnit = 0
Get-ChildItem -Path "crates" -Recurse -Filter "*.rs" | ForEach-Object {
    $matches = Select-String -Path $_.FullName -Pattern '#\[test\]' -AllMatches
    if ($matches) {
        foreach ($m in $matches) {
            $rustUnit += $m.Matches.Count
        }
    }
}

$pyCollect = & $PY -m pytest --collect-only -q 2>&1 | Out-String
$pyTests = 0
if ($pyCollect -match '(\d+)\s+tests?\s+collected') {
    $pyTests = [int]$Matches[1]
}

Write-Host "Rust #[test] in crates/: $rustUnit"
Write-Host "Python tests (pytest --collect-only): $pyTests"
Write-Host "Run .\scripts\ci_local.ps1 for the full gate."
