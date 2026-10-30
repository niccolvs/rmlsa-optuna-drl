# Crea .venv (Python 3.11) e instala todo. Uso, desde la raíz del repo:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Requiere uv: winget install astral-sh.uv
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv venv --python 3.11 .venv
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path

# PyTorch CPU (~200 MB). Para PPO con MlpPolicy la CPU suele ser igual o más rápida que la GPU.
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -e .
# DREAM-ON GYM: se clona en external/ e instala editable porque el paquete de
# PyPI/git omite bitRates.schema.json. --no-deps evita tensorflow==2.15 y mpi4py (no se usan).
if (-not (Test-Path external\dream-on-gym)) { git clone https://gitlab.com/IRO-Team/dream-on-gym-v2.0.git external\dream-on-gym }
uv pip install --no-deps -e external\dream-on-gym

Write-Host "`nListo. Activa con: .venv\Scripts\Activate.ps1  y prueba con: python scripts\smoke_test.py"
