#!/usr/bin/env bash
# Crea .venv (Python 3.11) e instala todo. Uso: bash scripts/setup.sh
# Requiere uv: curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv
export VIRTUAL_ENV="$PWD/.venv"

# PyTorch CPU. Para PPO con MlpPolicy la CPU suele ser igual o más rápida que la GPU.
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -e .
# DREAM-ON GYM: se clona en external/ e instala editable porque el paquete de
# PyPI/git omite bitRates.schema.json. --no-deps evita tensorflow==2.15 y mpi4py (no se usan).
[ -d external/dream-on-gym ] || git clone https://gitlab.com/IRO-Team/dream-on-gym-v2.0.git external/dream-on-gym
uv pip install --no-deps -e external/dream-on-gym

echo "Listo. Activa con: source .venv/bin/activate  y prueba con: python scripts/smoke_test.py"
