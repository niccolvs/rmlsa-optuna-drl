# Optimización de hiperparámetros con Optuna para RMLSA con DRL

Proyecto del curso **TEL 317** (USM, 2026) — Andrea Alfonzo, Nicolás Alvarez, Lorenzo Gonzalez.

Un agente de aprendizaje por refuerzo profundo (PPO, Stable-Baselines3) resuelve el problema
dinámico de ruteo, modulación y asignación de espectro (RMLSA) en redes ópticas elásticas sobre
[DREAM-ON GYM](https://gitlab.com/IRO-Team/dream-on-gym-v2.0). Sus hiperparámetros (tasa de
aprendizaje, factor de descuento, tamaño de lote) se ajustan con [Optuna](https://optuna.org)
para minimizar la **probabilidad de bloqueo**, y se compara con el agente por defecto y con la
heurística **SPFF** (Shortest-Path First-Fit).

## Instalación (~2 min, ~700 MB)

Requisitos: Git y [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`). uv descarga
Python 3.11 por su cuenta.

```powershell
git clone https://github.com/niccolvs/rmlsa-optuna-drl.git
cd rmlsa-optuna-drl
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Linux/macOS: bash scripts/setup.sh
.venv\Scripts\Activate.ps1
python scripts\smoke_test.py                                  # SPFF + PPO + Optuna en ~1-2 min
```

No hace falta CUDA, TensorFlow, MPI ni Visual Studio Build Tools aunque el README de DREAM-ON GYM
los pida: ver [docs/instalacion.md](docs/instalacion.md).

## Estructura

```
data/
  topologies/     NSFNet (14 nodos, 42 enlaces, 100 slots) y sus 3 rutas candidatas por par
  bitrates/       10/40/100/400 Gbps con BPSK…64-QAM (slots y alcance, tablas de Flex Net Sim)
src/rmlsa/
  env.py          RMLSAEnv (Gymnasium): acción = ruta, modulación por alcance, espectro First-Fit
  heuristics.py   allocators: first_fit (ruta del agente) y spff
scripts/
  setup.ps1/.sh             instalación
  smoke_test.py             prueba de punta a punta (~30 s)
  baseline_spff.py          bloqueo de SPFF por carga y semilla (+ gráfico)
  train_ppo.py              entrena PPO con hiperparámetros por defecto (+ TensorBoard)
docs/             notas (instalación, escenario, decisiones)
evidencias/       capturas y videos de los experimentos (versionado)
results/          modelos, estudios Optuna y logs (no versionado)
```

## Plan de trabajo

- [x] Entorno instalable y estructura del repositorio
- [x] Escenario y agente base: observación (par, bitrate y estado del espectro en las k rutas), episodios de 1000 solicitudes
- [x] Baseline SPFF (4 cargas × 5 semillas) y PPO por defecto (ver `evidencias/`)
- [ ] Estudio Optuna (TPE + pruning) con almacenamiento SQLite: `scripts/optuna_ppo.py`
- [ ] Evaluación final con múltiples semillas e importancia de hiperparámetros

## Créditos

Topología, rutas y bitrates en `data/` provienen de DREAM-ON GYM v2 (N. Jara, H. Pempelfort et al.,
SIMULTECH 2024), distribuido bajo GPL-3.0.
