# Optimización de hiperparámetros con Optuna para RMLSA con DRL

TEL-317 Redes Ópticas WDM (USM, 2026). Andrea Alfonzo, Nicolás Alvarez, Lorenzo Gonzalez.

Agente PPO (Stable-Baselines3) que elige la ruta de cada solicitud en RMLSA dinámico sobre NSFNet,
usando el simulador de [DREAM-ON GYM](https://gitlab.com/IRO-Team/dream-on-gym-v2.0). Con
[Optuna](https://optuna.org) se ajustan learning rate, gamma y batch size para minimizar la
probabilidad de bloqueo, y se compara contra PPO por defecto y SPFF.

## Instalación

Requiere Git y [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`).

```powershell
git clone https://github.com/niccolvs/rmlsa-optuna-drl.git
cd rmlsa-optuna-drl
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Linux/macOS: bash scripts/setup.sh
.venv\Scripts\Activate.ps1
python scripts\smoke_test.py
```

No hace falta CUDA, TensorFlow ni MPI. Detalles en [docs/instalacion.md](docs/instalacion.md).

## Uso

```powershell
python scripts\baseline_spff.py                                                   # bloqueo de SPFF
python scripts\train_ppo.py                                                       # PPO por defecto
python scripts\optuna_ppo.py --study-name ppo_rmlsa_100erl_v4 --n-trials 300      # estudio Optuna (~8 h)
python scripts\evaluar_final.py --study-name ppo_rmlsa_100erl_v4                  # defecto vs Optuna vs SPFF
python scripts\analisis_red.py --run ppo_rmlsa_100erl_v4                          # gráficos de la red
python scripts\maqueta_red.py                                                     # maqueta en vivo
python scripts\maqueta_red.py --video results\maqueta.mp4 --carga 150             # maqueta en video
```

`results/` ya trae el estudio final (base de Optuna, gráficos y los 6 modelos), así que la maqueta y
los gráficos se pueden correr sin volver a entrenar.

## Resultados

300 trials (90 completos, 210 podados). Mejor configuración: gamma 0.80, learning rate 1.8e-5,
batch size 128 (por defecto: 0.99, 3e-4, 64).

Evaluado con 10 cargas (25 a 250 Erlang) y 10 semillas de tráfico nuevas, PPO Optuna bloquea entre
11 % y 81 % menos que SPFF y entre 6 % y 47 % menos que PPO por defecto.

| Erlang | SPFF | PPO defecto | PPO Optuna |
|---|---|---|---|
| 25 | 0.0107 | 0.0038 | 0.0020 |
| 100 | 0.0985 | 0.0891 | 0.0798 |
| 250 | 0.2057 | 0.1955 | 0.1821 |

Capturas y datos de cada etapa en [evidencias/](evidencias/README.md). Informe y presentación en `entregas/`.

## Estructura

```
data/        NSFNet, rutas candidatas y bitrates
src/rmlsa/   entorno Gymnasium (env.py) y heurísticas (heuristics.py)
scripts/     instalación, SPFF, PPO, Optuna, evaluación, análisis y maqueta
docs/        instalación y escenario de simulación
evidencias/  capturas, videos y CSV de los experimentos
results/     estudio final (Optuna, evaluación y modelos)
entregas/    Taller 4 y entrega final
```

## Créditos

Topología, rutas y bitrates de `data/` vienen de DREAM-ON GYM v2 (N. Jara, H. Pempelfort et al.,
SIMULTECH 2024), licencia GPL-3.0.
