# Evidencias

Capturas, videos y datos de cada etapa, en carpetas `AAAA-MM-DD_etapaN`.

## 2026-11-03 — Etapa 1: referencia SPFF y agente base (objetivo específico 1)

Escenario: NSFNet, 100 slots por enlace, 3 rutas candidatas, bitrates 10–400 Gbps
(ver `docs/escenario.md`).

**SPFF** (`scripts/baseline_spff.py`, 10 000 solicitudes × 5 semillas):

| Carga [Erlang] | Bloqueo medio | Desv. estándar |
|----------------|---------------|----------------|
| 50             | 0.0382        | 0.0030         |
| 100            | 0.0950        | 0.0039         |
| 150            | 0.1372        | 0.0045         |
| 200            | 0.1732        | 0.0053         |

**PPO con hiperparámetros por defecto** (`scripts/train_ppo.py`, 100 Erlang, 200 000 pasos, 4.7 min):

| Agente        | Bloqueo (10 000 solicitudes, semilla 1000) |
|---------------|--------------------------------------------|
| PPO (defecto) | 0.0936                                     |
| SPFF          | 0.0938                                     |

La curva de bloqueo baja de ~0.20 a ~0.10 en los primeros ~40 000 pasos y después queda plana.
Con los valores por defecto, el agente aprende a rendir igual que SPFF, pero no mejor.

| Archivo                               | Contenido                                        |
|---------------------------------------|--------------------------------------------------|
| `01_spff_tabla_cargas.png`            | Salida de baseline_spff.py                       |
| `02_spff_grafico_cargas.png` / `.csv` | Bloqueo de SPFF vs carga                         |
| `03_ppo_default_entrenando.mp4`       | Entrenamiento en curso                           |
| `04_ppo_default_resultado.png`        | Evaluación final PPO vs SPFF                     |
| `05_ppo_default_curva_bloqueo.png`    | TensorBoard: `rmlsa/bloqueo_episodio`            |
| `06_ppo_default_curva_recompensa.png` | TensorBoard: `rollout/ep_rew_mean`               |

## 2026-11-03 — Etapa 2: estudio Optuna (objetivos específicos 2 y 3)

`scripts/optuna_ppo.py`: 30 trials, TPE (semilla 0) + MedianPruner, 50 000 pasos por trial en 5 etapas,
100 Erlang. Bloqueo medido con 3000 solicitudes (semilla de tráfico 1000). Duración: 18.5 min.

Espacio de búsqueda: `learning_rate` ∈ [1e-5, 1e-2] (log), `gamma` ∈ [0.8, 0.9999],
`batch_size` ∈ {32, 64, 128, 256, 512}.

| Resultado                  | Valor                                                   |
|----------------------------|---------------------------------------------------------|
| Trials completos / podados | 16 / 14                                                 |
| Mejor bloqueo (trial 27)   | 0.0763                                                  |
| Mejores hiperparámetros    | learning_rate = 4.82e-4, gamma = 0.987, batch_size = 512 |
| Importancia (PED-ANOVA)    | gamma 0.92, learning_rate 0.05, batch_size 0.03         |

Preliminar: el 0.0763 se midió con una sola semilla y 3000 solicitudes, y es el mejor de 30
intentos, así que puede estar algo favorecido por el azar. La comparación definitiva contra PPO
por defecto y SPFF (varias semillas y cargas) corresponde al objetivo 4.

| Archivo                            | Contenido                                     |
|------------------------------------|-----------------------------------------------|
| `01_optuna_terminal_30_trials.png` | Salida completa del estudio                   |
| `02_optuna_historial.png`          | Bloqueo por trial y mejor valor acumulado     |
| `03_optuna_importancia.png`        | Importancia de cada hiperparámetro            |
| `04_optuna_trials.csv`             | Todos los trials (parámetros, estado, valor)  |

## 2026-11-07 — Etapa 3: Optuna v2 y evaluación final (objetivos específicos 3, 4 y 5)

**Optuna v2** (`scripts/optuna_ppo.py`, estudio `ppo_rmlsa_100erl_v2`): 40 trials, TPE (semilla 0) +
MedianPruner, 100 000 pasos por trial en 5 etapas, 100 Erlang, bloqueo medido con 10 000 solicitudes
(semilla de tráfico 1000). Duración: 41 min.

| Resultado                  | Valor                                                     |
|----------------------------|-----------------------------------------------------------|
| Trials completos / podados | 26 / 14                                                   |
| Mejor bloqueo (trial 37)   | 0.0726                                                    |
| Mejores hiperparámetros    | learning_rate = 4.13e-5, gamma = 0.830, batch_size = 256  |
| Importancia (PED-ANOVA)    | gamma 0.54, learning_rate 0.44, batch_size 0.02           |

**Evaluación final** (`scripts/evaluar_final.py`): PPO por defecto y PPO con los hiperparámetros de
Optuna, cada uno entrenado con 3 semillas (0, 1, 2) durante 300 000 pasos a 100 Erlang. Evaluados con
10 000 solicitudes × 5 semillas de tráfico nuevas (2000–2004) por carga. Duración: 23.7 min.

| Carga [Erlang] | SPFF            | PPO defecto     | PPO Optuna          | Mejora vs SPFF |
|----------------|-----------------|-----------------|---------------------|----------------|
| 50             | 0.0400 ± 0.0020 | 0.0281 ± 0.0036 | **0.0207 ± 0.0015** | 48.3 %         |
| 100            | 0.1002 ± 0.0017 | 0.0895 ± 0.0048 | **0.0803 ± 0.0039** | 19.9 %         |
| 150            | 0.1430 ± 0.0051 | 0.1335 ± 0.0043 | **0.1228 ± 0.0044** | 14.1 %         |
| 200            | 0.1770 ± 0.0046 | 0.1665 ± 0.0056 | **0.1564 ± 0.0039** | 11.7 %         |

(media ± desviación estándar sobre 5 semillas de tráfico; para los PPO, además, sobre 3 entrenamientos)

El agente optimizado bloquea menos que los otros dos en todas las cargas, incluidas 50, 150 y 200
Erlang, con las que no fue entrenado. Sus 3 entrenamientos dan resultados casi iguales (0.0797–0.0807
a 100 Erlang), mientras que los del agente por defecto varían más (0.0859–0.0938).

| Archivo                         | Contenido                                       |
|---------------------------------|-------------------------------------------------|
| `01_optuna_v2_historial.png`    | Bloqueo por trial y mejor valor acumulado       |
| `02_optuna_v2_importancia.png`  | Importancia de cada hiperparámetro              |
| `03_optuna_v2_trials.csv`       | Todos los trials (parámetros, estado, valor)    |
| `04_final_bloqueo_vs_carga.png` | Bloqueo vs carga de los tres agentes            |
| `05_final_terminal_tabla.png`   | Salida de evaluar_final.py                      |
| `06_final_resumen.csv`          | Media y desviación por agente y carga           |
| `07_final_resultados.csv`       | Cada evaluación (agente, semilla, carga)        |
