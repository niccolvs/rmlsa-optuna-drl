#!/usr/bin/env bash
# Instala el entorno. Requiere uv.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv
export VIRTUAL_ENV="$PWD/.venv"

uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -e .
# DREAM-ON GYM editable (al paquete de pip le falta bitRates.schema.json)
[ -d external/dream-on-gym ] || git clone https://gitlab.com/IRO-Team/dream-on-gym-v2.0.git external/dream-on-gym
uv pip install --no-deps -e external/dream-on-gym

echo "Listo. Activa con: source .venv/bin/activate  y prueba con: python scripts/smoke_test.py"
