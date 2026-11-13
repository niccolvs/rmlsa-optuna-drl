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

## Etapa 4 (2026-11-08/09): Optuna v3, evaluación ampliada y análisis de la red

Carpeta: `2026-11-09_etapa4/`

**Optuna v3** (`scripts/optuna_ppo.py`): 100 trials, 200 000 pasos por trial a 100 Erlang, 5 etapas con
MedianPruner, cada medición promediada sobre 3 semillas de tráfico (1000, 1001, 1002). Se corrió en dos
sesiones (el estudio se reanuda desde la base SQLite); tiempo de cómputo total: 176.9 min.

| Resultado                  | Valor                                                     |
|----------------------------|-----------------------------------------------------------|
| Trials                     | 100: 40 completos, 60 podados                             |
| Mejor bloqueo              | 0.0756                                                    |
| Mejores hiperparámetros    | learning_rate = 2.16e-5, gamma = 0.836, batch_size = 64   |
| Importancia (PED-ANOVA)    | learning_rate 0.716, gamma 0.189, batch_size 0.095        |

**Evaluación ampliada** (`scripts/evaluar_final.py`): 3 entrenamientos (semillas 0, 1, 2) de 300 000 pasos
por configuración, 10 cargas × 10 semillas de tráfico nuevas (2000–2009) × 10 000 solicitudes. 59.6 min.

| Carga [Erlang] | SPFF            | PPO defecto     | PPO Optuna          | Optuna vs SPFF | Optuna vs defecto |
|----------------|-----------------|-----------------|---------------------|----------------|-------------------|
| 25             | 0.0107 ± 0.0009 | 0.0038 ± 0.0005 | **0.0023 ± 0.0004** | 78.4 %         | 39.2 %            |
| 50             | 0.0398 ± 0.0017 | 0.0279 ± 0.0009 | **0.0214 ± 0.0014** | 46.3 %         | 23.5 %            |
| 75             | 0.0718 ± 0.0017 | 0.0606 ± 0.0016 | **0.0525 ± 0.0014** | 26.9 %         | 13.4 %            |
| 100            | 0.0985 ± 0.0024 | 0.0891 ± 0.0019 | **0.0819 ± 0.0016** | 16.8 %         | 8.0 %             |
| 125            | 0.1202 ± 0.0022 | 0.1126 ± 0.0018 | **0.1045 ± 0.0019** | 13.0 %         | 7.2 %             |
| 150            | 0.1416 ± 0.0032 | 0.1328 ± 0.0018 | **0.1240 ± 0.0021** | 12.4 %         | 6.7 %             |
| 175            | 0.1591 ± 0.0031 | 0.1498 ± 0.0027 | **0.1408 ± 0.0025** | 11.5 %         | 6.0 %             |
| 200            | 0.1757 ± 0.0026 | 0.1662 ± 0.0026 | **0.1568 ± 0.0028** | 10.8 %         | 5.6 %             |
| 225            | 0.1912 ± 0.0025 | 0.1812 ± 0.0023 | **0.1704 ± 0.0025** | 10.9 %         | 6.0 %             |
| 250            | 0.2057 ± 0.0030 | 0.1955 ± 0.0021 | **0.1849 ± 0.0024** | 10.1 %         | 5.4 %             |

(media ± IC 95 % sobre 10 semillas de tráfico; para los PPO, cada semilla promedia los 3 entrenamientos.
PPO Optuna bloquea menos que SPFF y que PPO defecto en las 10 semillas de cada carga.)

**Análisis de la red a 100 Erlang**: PPO defecto manda ~30 % de las solicitudes por una ruta alternativa a
cualquier carga; PPO Optuna se desvía poco con la red vacía (2.4 % a 25 Erlang) y más a medida que se llena
(16.7 % a 250). El bloqueo se concentra en 400 Gbps (SPFF 35.2 %, PPO Optuna 32.0 %) y en pares lejanos
(> 3000 km: SPFF 19.2 %, PPO Optuna 16.3 %). PPO Optuna ocupa más espectro en promedio (31.4 % vs 27.3 %)
porque acepta más conexiones y algunas van por rutas más largas.

| Archivo                                   | Contenido                                               |
|-------------------------------------------|---------------------------------------------------------|
| `01_optuna_v3_inicio.png`                 | Inicio del estudio (primeros trials)                    |
| `02_optuna_v3_resumen.png`                | Resumen final de Optuna                                 |
| `03_evaluacion_tabla_y_mejoras.png`       | Tabla de bloqueo y mejoras con IC 95 %                  |
| `04_analisis_red_terminal.png`            | Rutas, bloqueo por velocidad y por distancia            |
| `05`–`10_optuna_*.png`                    | Historial, importancia, slice, coordenadas paralelas, valores intermedios, gamma vs learning rate |
| `11_optuna_trials.csv`                    | Los 100 trials                                          |
| `12_bloqueo_vs_carga.png`                 | Bloqueo vs carga con IC 95 % (escala log)               |
| `13_mejora_vs_carga.png`                  | Reducción relativa del bloqueo vs carga                 |
| `14_curvas_aprendizaje.png`               | Bloqueo por episodio durante el entrenamiento           |
| `15_uso_rutas.png`                        | % de solicitudes por ruta alternativa vs carga          |
| `16_bloqueo_por_bitrate.png`              | Bloqueo por velocidad pedida                            |
| `17_bloqueo_por_distancia.png`            | Bloqueo por largo de la ruta más corta del par          |
| `18_mapa_ocupacion.png`                   | NSFNet coloreada por ocupación de cada enlace           |
| `19_resumen.csv`, `20`–`22_mejora_*.csv`  | Medias, IC 95 % y mejoras por carga                     |
| `23_resultados.csv`                       | Cada evaluación (agente, entrenamiento, carga, semilla) |

## Maqueta en vivo (2026-11-13)

Carpeta: `2026-11-13_maqueta/`. Generada con `scripts/maqueta_red.py`: SPFF y PPO Optuna (modelo
`optuna_s0` de la evaluación v3) atienden la misma secuencia de solicitudes (semilla 2000) en NSFNet.
Gris: ocupación de cada enlace; rojo: solicitudes bloqueadas; celeste: aceptadas por una ruta alternativa.

| Archivo                  | Contenido                                                                   |
|--------------------------|-----------------------------------------------------------------------------|
| `01_maqueta_150erl.mp4`  | Video de 33 s, 6 000 solicitudes a 150 Erlang. Final: SPFF 14.1 %, PPO Optuna 12.5 % |
| `02_maqueta_cuadro.png`  | Un cuadro del video                                                         |

La ventana interactiva (`python scripts/maqueta_red.py`) permite cambiar la carga (25–250 Erlang), la
velocidad y el agente de la derecha (PPO Optuna o PPO por defecto).
