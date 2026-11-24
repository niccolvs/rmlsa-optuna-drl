# Contexto del proyecto (para retomar en cualquier PC, con o sin Claude Code)

Claude Code lee este archivo automáticamente al abrir el repo. También sirve como "estado del proyecto"
para cualquier integrante del grupo. Última actualización: 2026-11-22.

## Qué es

Proyecto del ramo **TEL 317 (USM, 2026)**. Integrantes: Andrea Alfonzo, Nicolás Alvarez y Lorenzo Gonzalez.

- **Objetivo general**: optimizar con Optuna los hiperparámetros (learning rate, gamma, batch size) de un
  agente PPO (Stable-Baselines3) que resuelve **RMLSA dinámico** en NSFNet, sobre DREAM-ON GYM. Se busca
  minimizar la **probabilidad de bloqueo** y comparar contra PPO por defecto y la heurística **SPFF**.
- La propuesta original (Taller 4, formato IEEE) está en `entregas/Taller4/` (informe + presentación).
  Ahí están los 5 objetivos específicos. **Los 5 están cumplidos y tienen evidencia** (ver tabla abajo).

## Estado al 24-11-2026  ⬅️ EMPEZAR AQUÍ

- **Parte experimental: CERRADA.** No hace falta volver a entrenar nada.
- **Presentación: miércoles 25-11-2026** (10 min) + **informe formato IEEE**. La pauta ya llegó:
  `entregas/final/Pauta_Taller_Final_2026.pdf`. Secciones: portada (proyecto, nombre del equipo, integrantes),
  problema, objetivos, metodología, demostración, resultados, autocrítica y trabajo futuro.
- **Todo el material para armarlos está en `entregas/final/GUION_presentacion_e_informe.md`**: plan de 12
  diapositivas con tiempos, qué decir en cada una, todos los números, autocrítica, preguntas probables con
  respuesta, referencias, estado del informe y qué imagen va en cada página.
- **Informe (Overleaf)**: hecho hasta Metodología (`entregas/final/informe_avance_hasta_metodologia.tex.txt`).
  Faltan IV Demostración, V Resultados, VI Autocrítica y trabajo futuro, VII Conclusiones. Forma de trabajo:
  Claude crea cada sección en un .txt, lo abre en el Bloc de notas, Nicolás copia/pega en Overleaf y avisa;
  se borra y se abre el siguiente. Estructura = la de la pauta del Taller Final, no la del Taller 5.
- **Pendiente decidir**: nombre del equipo para la portada; demo con video (recomendado) o ventana interactiva.
- Mientras tanto existe un **informe de estudio** (no es la entrega): `informe/Informe_TEL317_Optuna_RMLSA.pdf`
  (17 págs.). Explica todo para que el grupo entienda y pueda defender el trabajo: qué es RL/PPO,
  DREAM-ON GYM, Optuna (TPE, poda), escenario, semillas, resultados, limitaciones y 12 preguntas
  probables con respuesta. Se regenera con `python informe/informe_corto.py` y lee todos los números
  desde `results/` (no hay cifras escritas a mano).

## Resultados finales (estudio de 300 trials, `ppo_rmlsa_100erl_v4`)

La profesora sugirió ~300 trials, así que el estudio de 100 (v3) se extendió a 300 (v4 = copia de v3 +
200 trials más; v3 quedó intacto en la base).

| Dato | Valor |
|---|---|
| Trials | 300: 90 completos, 210 podados (MedianPruner) |
| Cómputo | 487 min de Optuna (suma de trials) + ~60 min de evaluación final, en CPU |
| Mejor trial | #210, bloqueo **0.0753** (con 100 trials era el #30, 0.0756) |
| Mejores hiperparámetros | **gamma 0.800**, **learning rate 1.81e-5**, batch size 128 (por defecto: 0.99, 3e-4, 64) |
| Importancia (PED-ANOVA) | gamma 0.64, batch size 0.20, learning rate 0.16 (con 100 trials: lr 0.72, gamma 0.19) |

Bloqueo (media ± IC 95 %, 10 semillas de tráfico nuevas 2000–2009, 3 entrenamientos por agente):

| Erlang | SPFF | PPO defecto | PPO Optuna | Optuna vs SPFF | Optuna vs defecto |
|---|---|---|---|---|---|
| 25 | 0.0107 | 0.0038 | **0.0020** | −81 % | −47 % |
| 50 | 0.0398 | 0.0279 | **0.0204** | −49 % | −27 % |
| 100 (carga de entrenamiento) | 0.0985 | 0.0891 | **0.0798** | −19 % | −10 % |
| 150 | 0.1416 | 0.1328 | **0.1230** | −13 % | −7 % |
| 250 | 0.2057 | 0.1955 | **0.1821** | −11 % | −7 % |

PPO Optuna gana a SPFF y a PPO defecto en **las 10 semillas de cada una de las 10 cargas**.
Tabla completa e interpretación: `evidencias/README.md` (sección 2026-11-20).

**Por qué gana** (análisis de red): PPO defecto se desvía de la ruta más corta ~30 % de las veces sin
importar la carga; PPO Optuna aprendió *cuándo* desviarse (1 % con la red vacía, 17 % saturada). El bloqueo
se concentra en conexiones de 400 Gbps y pares lejanos (> 3000 km), y ahí es donde PPO Optuna mejora más.

**Mensajes clave para la defensa**
- Triplicar la búsqueda (100 → 300) mejoró el mejor trial en solo 0.0003 (menos que el ruido entre
  semillas) y en la misma zona → el óptimo es robusto.
- Lo estable entre estudios es **dónde** están los buenos trials: gamma ≈ 0.80–0.84 y lr ≈ 2e-5. El orden
  de importancia cambia según los trials explorados (TPE concentra la búsqueda), por eso no conviene
  venderlo como absoluto.
- ⚠️ Pregunta probable: el mejor gamma (0.800) quedó en el **borde inferior** del rango buscado
  (0.8–0.9999). Respuesta: es una limitación; un trabajo futuro debería abrir el rango hacia abajo.
- Poda: ahorró ~5.6 h en el estudio de 300.
- En la maqueta/mapa, el enlace largo Seattle(0)–Champaign(7) pasa visualmente bajo Lincoln(6): es un solo
  enlace de 2840 km, no dos.

## Historia de las etapas (todas con capturas y datos en `evidencias/`)

| Fecha | Etapa | Resultado | Objetivo |
|---|---|---|---|
| 03-11 | Etapa 1: escenario, SPFF y PPO por defecto | SPFF 100 Erl = 0.095; PPO defecto ≈ SPFF | 1 |
| 03-11 | Etapa 2: Optuna v1 exploratorio (30 trials) | 0.0763 | 2, 3 |
| 07-11 | Etapa 3: Optuna v2 (40 trials) + evaluación 4 cargas × 5 semillas | 0.0726; medición con 1 semilla era ruidosa | 3, 4 |
| 09-11 | Etapa 4: Optuna v3 (100 trials, 3 semillas por trial) + evaluación 10 cargas × 10 semillas + análisis de red | 0.0756 | 3, 4, 5 |
| 13-11 | Maqueta en vivo (`scripts/maqueta_red.py`) | video para la presentación | — |
| 20-11 | **Estudio final de 300 trials (v4)** + evaluación + análisis + maqueta | 0.0753 | 3, 4, 5 |

Lección que vale la pena contar: v1/v2 medían cada trial con 1 semilla y las conclusiones se invertían
entre estudios; desde v3 cada trial se mide con 3 semillas (1000–1002).

## Decisiones de diseño (y por qué)

- **`RMLSAEnv` propio** (`src/rmlsa/env.py`): el entorno de DREAM-ON GYM v2 (`RlOnEnv`/`simulator_finite`)
  está roto o es un esqueleto (observación constante, episodios de 1 paso). Se usa su topología, rutas y
  bitrates, con nuestro entorno Gymnasium encima. Acción = cuál de 3 rutas; modulación por alcance;
  espectro First-Fit. Observación de 47 valores; recompensa ±1; episodios de 1000 solicitudes; warm-up
  de 3000 con SPFF. Detalle en `docs/escenario.md`.
- **Semillas**: entrenamiento 0; Optuna mide con 1000–1002; evaluación final con 2000–2009 (tráfico que
  Optuna nunca vio); entrenamientos finales con semillas 0, 1 y 2.
- **Optuna**: TPESampler(seed=0), MedianPruner(n_startup_trials=5, n_warmup_steps=1), 200 000 pasos por
  trial en 5 etapas, base SQLite `results/optuna.db` (si se corta con Ctrl+C se reanuda; los trials FAIL
  no cuentan como hechos).
- **Entrenamiento de 1M pasos descartado**: las curvas de aprendizaje se aplanan antes de 150 000 pasos.
- **Se quitaron KSP-FF y optuna-dashboard** para ajustarse a lo planteado en la propuesta.
- **Workshop multi-agente con LLM** (Jesús Chaffe, repo YizzHub/workshop-ia-multiagentica-redes-de-acceso):
  solo se menciona como discusión/trabajo futuro (un orquestador que reciba instrucciones en lenguaje
  natural y use el agente como herramienta). **No se integra.**
- Trabajo futuro mencionado en la propuesta: QKD (canales clásicos + cuánticos).

## Cómo retomar en otro PC

```powershell
git clone https://github.com/niccolvs/rmlsa-optuna-drl.git
cd rmlsa-optuna-drl
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # ~2 min; requiere uv (winget install astral-sh.uv)
.venv\Scripts\Activate.ps1
```

El repo **incluye una copia del estudio final** (excepción en `.gitignore`), así que no hace falta
re-entrenar (~10 h):
- `results/optuna.db`: todos los estudios Optuna (v1, v2, v3 y v4).
- `results/optuna/ppo_rmlsa_100erl_v4/` y `results/final/ppo_rmlsa_100erl_v4/`: gráficos y CSV.
- `results/models/final/ppo_rmlsa_100erl_v4/`: los 6 modelos finales (los usa la maqueta).

Comandos útiles:

```powershell
python scripts\maqueta_red.py                      # ventana interactiva (sliders de carga, velocidad, agente)
python scripts\maqueta_red.py --video results\maqueta\maqueta_150erl_v4.mp4 --carga 150   # video MP4
pip install reportlab                              # una vez, solo para el informe
python informe\informe_corto.py                    # -> informe\Informe_TEL317_Optuna_RMLSA.pdf
python scripts\optuna_ppo.py --study-name ppo_rmlsa_100erl_v4 --solo-graficos   # regenerar gráficos Optuna
python scripts\analisis_red.py --run ppo_rmlsa_100erl_v4                       # regenerar gráficos de red
```

Cómo se corrió el estudio final (por si hay que repetirlo; ~10 h en total):

```
python scripts\optuna_ppo.py --study-name ppo_rmlsa_100erl_v4 --copiar-de ppo_rmlsa_100erl_v3 --n-trials 300
python scripts\evaluar_final.py --study-name ppo_rmlsa_100erl_v4
python scripts\analisis_red.py --run ppo_rmlsa_100erl_v4
python scripts\maqueta_red.py --video results\maqueta\maqueta_150erl_v4.mp4 --carga 150
```

Ojo: `evaluar_final.py`, `analisis_red.py` y `optuna_ppo.py` tienen v3 como valor por defecto: pasar
siempre el nombre del estudio v4.

## Problemas conocidos / gotchas

- DREAM-ON GYM declara tensorflow==2.15 y mpi4py pero no los usa → se instala con `--no-deps`. El paquete
  de pip omite `bitRates.schema.json` → se instala editable desde el clon en `external/` (lo hace
  `setup.ps1`). Con gymnasium ≥ 1.0 hay que usar `.unwrapped`. Ver `docs/instalacion.md`.
- GPU no ayuda (el cuello de botella es el simulador en Python, un paso a la vez). Todo corre en CPU.
- **cmd vs PowerShell**: en PowerShell no existe `&&`; usar `;` o correr en cmd.
- El informe usa las fuentes Arial/Consolas de `C:\Windows\Fonts` (pensado para Windows).
- `maqueta_red.py` usa `imageio-ffmpeg` (ya en las dependencias) para el MP4 y
  `data/mapas/us_states.json` para dibujar el mapa de EE. UU.; los nodos están en sus ciudades reales
  (0 Seattle, 1 Palo Alto, 2 San Diego, 3 Salt Lake City, 4 Boulder, 5 Houston, 6 Lincoln, 7 Champaign,
  8 Pittsburgh, 9 Atlanta, 10 Ann Arbor, 11 Ithaca, 12 Princeton, 13 College Park).

## Material en el repo

| Ruta | Qué es |
|---|---|
| `entregas/Taller4/` | Propuesta (informe IEEE + presentación) entregada el 29-10 |
| `informe/` | Informe de estudio (PDF) y el script que lo genera (`informe_corto.py`, `figuras.py`, `fig/`) |
| `docs/Guia_DREAM-ON-GYM_y_Optuna.pdf` | Guía introductoria de DREAM-ON GYM y Optuna |
| `docs/escenario.md`, `docs/instalacion.md` | Escenario de simulación e instalación |
| `evidencias/` | Capturas, videos y datos de cada etapa (con su `README.md`) |

## Forma de trabajo con Claude (preferencias de Nicolás)

- Los entrenamientos largos los corre Nicolás en **cmd**, con instrucciones paso a paso (clic por clic) y
  avisos de cuándo sacar capturas para `evidencias/`. Claude se encarga del código, instalaciones,
  ordenar evidencias, READMEs y subir a GitHub.
- Explicar todo en el chat, en español y sin tecnicismos innecesarios.
- No agregar herramientas ni "extras" fuera de lo planteado en la propuesta (se nota hecho por IA).
- Commits con identidad local `niccolvs`. Repo privado hasta terminar el ramo (después, público como
  portafolio).

## Próximos pasos

1. Terminar el informe IEEE: IV Demostración, V Resultados, VI Autocrítica y trabajo futuro, VII Conclusiones
   (contenido y números en `entregas/final/GUION_presentacion_e_informe.md`).
2. Armar la presentación del 25-11 (12 páginas, imágenes indicadas en el guion; video
   `evidencias/2026-11-20_300trials/23_maqueta_150erl.mp4`).
3. Definir nombre del equipo para la portada.
