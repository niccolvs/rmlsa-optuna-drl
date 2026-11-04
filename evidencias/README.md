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
