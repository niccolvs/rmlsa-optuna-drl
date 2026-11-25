# Instala el entorno. Requiere uv.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv venv --python 3.11 .venv
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path

uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -e .
# DREAM-ON GYM editable (al paquete de pip le falta bitRates.schema.json)
if (-not (Test-Path external\dream-on-gym)) { git clone https://gitlab.com/IRO-Team/dream-on-gym-v2.0.git external\dream-on-gym }
uv pip install --no-deps -e external\dream-on-gym

Write-Host "`nListo. Activa con: .venv\Scripts\Activate.ps1  y prueba con: python scripts\smoke_test.py"
