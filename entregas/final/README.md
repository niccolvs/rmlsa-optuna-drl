# Taller Final TEL-317: entrega y guía de estudio

Entrega del 25-11-2026: presentación de 10 minutos + informe en formato IEEE
(pauta: `Pauta_Taller_Final_2026.pdf`). Esta carpeta tiene lo entregado y lo necesario para volver a
estudiar el proyecto más adelante (certamen, examen o defensa del portafolio).

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `Informe_final_TEL317.pdf` | **Informe entregado** (IEEE, 6 páginas, 10 referencias). Escrito en Overleaf. |
| `Presentacion_final_TEL317.pdf` | **Presentación entregada** (Canva, 13 láminas). En el PDF la demostración es una imagen fija; el video está en `evidencias/2026-11-20_300trials/23_maqueta_150erl.mp4`. |
| `references.bib` | Bibliografía del informe (10 entradas). |
| `figuras_informe/` | Las 5 imágenes del informe con los nombres que espera el `.tex` (más `nsfnet.png`, que quedó opcional y no se usó). |
| `GUION_presentacion_e_informe.md` | **El material de estudio más completo**: qué decir en cada lámina, todos los números, autocrítica, preguntas probables con respuesta y referencias. |
| `Pauta_Taller_Final_2026.pdf` | Pauta de la profesora. |
| `borradores/` | Textos intermedios que se pegaron en Overleaf y Canva (el informe por partes, los textos de las láminas). Sirven para ver cómo se construyó; la versión final es el PDF. |

El código fuente LaTeX vive en Overleaf. Para guardarlo aquí: Overleaf → Menu → Download → Source,
y descomprimir en `entregas/final/latex/`.

## Cómo estudiar (en este orden)

1. **`Informe_final_TEL317.pdf`** (20 min): la versión corta y oficial de todo.
2. **`informe/Informe_TEL317_Optuna_RMLSA.pdf`** (17 págs.): informe de estudio que explica desde cero
   RL, PPO, DREAM-ON GYM, Optuna (TPE y poda), el escenario, las semillas y las limitaciones.
3. **`GUION_presentacion_e_informe.md`**, sección 8: las preguntas probables con su respuesta.
   Practicar respondiéndolas en voz alta sin mirar.
4. Si hay tiempo: `evidencias/README.md` (historia de cada etapa con sus capturas) y el video de la maqueta.

## El proyecto en 10 ideas

1. **Problema:** en una red óptica elástica (EON) cada conexión pide *slots* de espectro contiguos y
   continuos en toda su ruta. En RMLSA dinámico hay que elegir ruta, modulación y *slots* al momento;
   si no cabe, la solicitud se **bloquea**. Métrica: probabilidad de bloqueo.
2. **Heurística de referencia:** SPFF usa siempre la ruta más corta y el primer bloque libre, aunque
   esa ruta esté llena.
3. **Agente:** PPO (Stable-Baselines3) que **solo elige la ruta** entre 3 candidatas. Modulación y
   First-Fit son iguales a SPFF, así que la comparación aísla el valor de la decisión de ruta.
   Observación de 47 valores; recompensa +1 aceptada / −1 bloqueada.
4. **Entorno propio:** el entorno Gymnasium de DREAM-ON GYM 0.0.9 estaba roto; se usó su simulador,
   topología (NSFNet) y rutas con una capa Gymnasium propia (`src/rmlsa/env.py`).
5. **Optuna:** busca learning rate (1e-5 a 1e-2, log), gamma (0,8 a 0,9999) y tamaño de lote
   (32 a 512). TPE propone combinaciones parecidas a las que funcionaron; la poda (MedianPruner)
   corta los trials que van peor que la mediana. Cada trial se mide con 3 semillas de tráfico.
6. **Resultado de Optuna:** 300 trials (90 completos, 210 podados; la poda ahorró ~5,6 h). Mejor:
   **gamma 0,80, lr 1,8e-5, lote 128** (por defecto: 0,99, 3e-4, 64). Gamma fue el más influyente.
7. **Evaluación honesta:** semillas separadas (entrenamiento 0–2, Optuna 1000–1002, prueba final
   2000–2009 que nadie vio), 10 cargas × 10 semillas, IC 95 %.
8. **Bloqueo:** PPO Optuna gana en los 100 casos. Frente a SPFF baja el bloqueo entre 11 % y 81 %;
   frente a PPO por defecto, entre 6 % y 47 %. A 100 Erlang: SPFF 0,0985 → Optuna 0,0798.
9. **Por qué gana:** PPO por defecto se desvía de la ruta corta ~30 % de las veces siempre; PPO Optuna
   aprendió **cuándo** desviarse (1 % con la red vacía, 17 % saturada). Mejora más en 400 Gbps y en
   pares lejanos.
10. **Por qué gamma bajo:** el horizonte del agente es ~1/(1−γ) solicitudes: ~100 con 0,99 y ~5 con
    0,80. Como el tráfico llega al azar, mirar muy lejos solo mete ruido.

**Autocrítica en una línea:** gamma quedó en el borde del rango buscado, una sola topología y una
carga de entrenamiento, el agente solo elige la ruta, y solo se comparó con SPFF.

## Detalles conocidos de lo entregado

Para no confundirse al releer:

- **Lámina 6** de la presentación: el gráfico de gamma muestra 0,99, 0,987 y 0,83 (los mejores de
  estudios anteriores); el valor final es 0,80. Se dejó así a propósito como ilustración del horizonte.
- **Informe, página 4:** queda un espacio en blanco en la columna izquierda porque la Tabla V y las
  Fig. 4 y 5 se fueron a la página 5. Es solo diagramación (se arregla con `[!htbp]`).
- **Informe:** quedaron detalles menores de plantilla y ortografía ("1st" en los autores, "(Poda)",
  "resolviendo los 5 objetivos", "Stable-baselines3" en la referencia [5]; el `references.bib` de
  esta carpeta ya tiene esa mayúscula corregida).
- La pauta pedía el **nombre del equipo** en la portada; la presentación entregada no lo tiene.
- Las cifras del informe se recalcularon desde `results/final/ppo_rmlsa_100erl_v4/` y coinciden.
