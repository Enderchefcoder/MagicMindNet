# Resolve repo Python: prefer .venv when present (Windows).
$root = Split-Path $PSScriptRoot -Parent
$venvPy = Join-Path (Join-Path $root ".venv") "Scripts\python.exe"
if (Test-Path $venvPy) {
    (Resolve-Path $venvPy).Path
} else {
    "python"
}
