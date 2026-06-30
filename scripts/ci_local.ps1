# Local CI mirror — run from repo root (uses .venv Python when present).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$PY = & "$PSScriptRoot\venv_python.ps1"

function Invoke-Step([string]$Label, [scriptblock]$Body) {
    Write-Host "== $Label =="
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Body 2>&1 | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
        if ($null -ne $code -and $code -ne 0) {
            exit $code
        }
    } finally {
        $ErrorActionPreference = $prev
    }
}

Invoke-Step "cargo test" { cargo test --workspace }
Invoke-Step "maturin develop" { maturin develop --release -m crates/mmn-py/Cargo.toml }
Invoke-Step "pytest" { & $PY -m pytest -q }
Invoke-Step "ruff" {
    if (Get-Command ruff -ErrorAction SilentlyContinue) {
        ruff check python tests examples
    } else {
        Write-Host "skip ruff (pip install -e '.[dev]')"
    }
}
Invoke-Step "quickstart" { & $PY examples/quickstart.py }
Invoke-Step "examples smoke" { .\scripts\smoke_examples.ps1 }

Write-Host "CI local: OK"
