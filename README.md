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
  bitrates/       10–1000 Gbps con BPSK/QPSK/8-QAM/16-QAM (slots y alcance, banda C)
src/rmlsa/
  env.py          make_env(): entorno DREAM-ON GYM con tráfico Poisson/exponencial y semillas
  heuristics.py   allocators: first_fit (ruta elegida por el agente) y spff
scripts/
  setup.ps1/.sh   instalación
  smoke_test.py   prueba de punta a punta
docs/             notas (instalación, decisiones)
notebooks/        análisis y gráficos
results/          modelos, estudios Optuna y logs (no versionado)
```

## Plan de trabajo

- [x] Entorno instalable y estructura del repositorio
- [ ] Agente base: diseñar la observación (estado del espectro en las k rutas) y la terminación del episodio
- [ ] Baseline SPFF y PPO por defecto bajo distintas cargas (Erlang) y semillas
- [ ] Estudio Optuna (TPE + pruning) con almacenamiento SQLite y `optuna-dashboard`
- [ ] Evaluación final con múltiples semillas e importancia de hiperparámetros

## Créditos

Topología, rutas y bitrates en `data/` provienen de DREAM-ON GYM v2 (N. Jara, H. Pempelfort et al.,
SIMULTECH 2024), distribuido bajo GPL-3.0.
