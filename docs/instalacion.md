# Notas de instalación (investigado 2026-10-29)

## Resumen

La instalación **no es pesada** si se ignoran las dependencias que DREAM-ON GYM declara pero no usa.
Medido en Windows 11: ~1.5 min y ~700 MB (Python 3.11 + PyTorch CPU + SB3 + Optuna).

## Qué pide DREAM-ON GYM y qué se necesita de verdad

| Lo que pide su README / setup.py           | ¿Se usa en el código? | Decisión                       |
|--------------------------------------------|-----------------------|--------------------------------|
| TensorFlow 2.15 (+ CUDA "septiembre 2018") | No (0 imports)        | Omitir (`--no-deps`)           |
| mpi4py + MS-MPI + VS Build Tools 2019      | No (0 imports)        | Omitir                         |
| Python 3.10.6                              | —                     | Python 3.11 funciona           |
| Stable-Baselines3 "desde git"              | Sí (ejemplos)         | `pip install stable-baselines3`|
| gymnasium 0.28.1                           | Sí                    | Versión actual funciona        |
| PyTorch                                    | Sí (vía SB3)          | Build CPU                      |

TensorFlow y MPI son restos de Stable-Baselines **v2** (PPO2), que DREAM-ON GYM usaba antes de
migrar a SB3. Instalar TensorFlow 2.15 además obliga a Python ≤ 3.11 y suma ~2 GB.

DREAM-ON GYM es **Python puro** (incluye una reimplementación en Python de Flex Net Sim,
`simNetPy`), así que no hay nada que compilar.

## Problemas conocidos

1. **El paquete omite `bitRates.schema.json`.** Instalado desde PyPI o `git+https`, falla al cargar
   cualquier topología (`FileNotFoundError ... bitRates.schema.json`). Solución: clonar el repo e
   instalar en modo editable (`pip install --no-deps -e external/dream-on-gym`), que es lo que hace
   `scripts/setup.*`.
2. **gymnasium ≥ 1.0 ya no reenvía atributos a través de wrappers.** Los ejemplos oficiales llaman
   `env.setRewardFunc(...)` sobre lo que devuelve `gymnasium.make`; hay que usar
   `gymnasium.make(...).unwrapped` (ya resuelto en `src/rmlsa/env.py`).
3. **El entorno es un esqueleto.** Por defecto la observación es constante y cada paso termina el
   episodio (`terminated=True`). Para que PPO aprenda algo hay que definir:
   - `setStateFunc`: observación (p. ej. ocupación de slots / bloques libres en las k rutas) y el
     `observation_space` correspondiente.
   - `setTerminatedFunc` / `setTruncatedFunc`: largo del episodio.
   Este es el trabajo real del proyecto, no la instalación.

## GPU

Hay una RTX 4050 (6 GB) disponible, pero para PPO con `MlpPolicy` la GPU no ayuda (el cuello de
botella es el simulador en Python, un paso a la vez). Lo que sí acelera: correr varios trials de
Optuna en paralelo (`n_jobs` o varios procesos contra el mismo SQLite).

## Costo de cómputo esperado

Con presupuesto por trial de ~50k pasos, el simulador en Python da del orden de cientos a pocos miles
de pasos/s, o sea minutos por trial. Un estudio de 50–100 trials son horas, no días, sobre todo con
pruning (`MedianPruner`) que corta los trials malos temprano. A medir con el smoke test.
