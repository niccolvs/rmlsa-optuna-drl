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
