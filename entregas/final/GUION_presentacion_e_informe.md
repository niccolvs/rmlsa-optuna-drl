# Taller Final TEL-317 — material base (presentación + informe IEEE)

Pauta (C:\Users\nicol\Downloads\Taller Final 2026.pdf, prof. Patricia Morales):
- Presentación de 10 min, mié 25-11-2026, "visual e interesante".
- Secciones: Portada (proyecto, NOMBRE DEL EQUIPO, integrantes) · Problema · Objetivos · Metodología · Demostración ·
  Resultados (puede ser >1 slide) · Autocrítica y trabajo futuro.
- Informe formato IEEE con la misma información.
- Taller 4 (propuesta) tenía: título "Optimización de hiperparámetros con Optuna mediante aprendizaje de refuerzo
  profundo en el problema de RMLSA", estilo Canva azul marino/celeste, 6 refs IEEE. Correos: andrea.alfonzo@usm.cl,
  nicolas.alvarezsa@usm.cl, lorenzo.gonzalez@usm.cl. Departamento de Electrónica, USM, Santiago.
- FALTA: nombre del equipo (no existe en ningún lado).

## Plan de diapositivas (~10 min)
| # | Slide | Tiempo |
|---|---|---|
| 1 | Portada | 0:15 |
| 2 | Problema: EON + RMLSA + bloqueo (figura espectro) | 1:15 |
| 3 | Por qué DRL + por qué Optuna (sensibilidad a hiperparámetros, Doherty 2025) | 0:45 |
| 4 | Objetivos (general + 5 específicos, con ✔) | 0:45 |
| 5 | Metodología 1: escenario y agente (NSFNet, MDP) | 1:15 |
| 6 | Metodología 2: Optuna (TPE, poda, espacio, semillas separadas) | 1:15 |
| 7 | Demostración: video maqueta 150 Erl | 1:00 |
| 8 | Resultados 1: Optuna (gamma vs lr, importancia, top trials) | 1:00 |
| 9 | Resultados 2: bloqueo vs carga + tabla | 1:00 |
| 10 | Resultados 3: por qué gana (uso de rutas, 400G/distancia, mapa) | 0:45 |
| 11 | Autocrítica y trabajo futuro | 0:45 |
| 12 | Conclusiones / gracias | 0:15 |

## 1. Problema
- Demanda crece ~40 %/año (dicho en Taller 4). WDM fijo (grilla 50 GHz) → EON: espectro en slots (FSU), cada
  conexión toma solo los que necesita.
- RMLSA: Routing (ruta), Modulation Level (bits/símbolo: más eficiente = menos slots pero menos alcance),
  Spectrum Assignment (qué slots). Restricciones: continuidad (mismos slots en todos los enlaces) y contigüidad
  (slots pegados). Fragmentación = huecos inutilizables.
- Dinámico: llegadas Poisson (λ), duración exponencial (1/μ = 1). Carga Erlang = λ/μ = conexiones activas promedio.
- Bloqueo: no hay bloque libre que cumpla → se rechaza. Métrica: probabilidad de bloqueo = bloqueadas/totales.
- Heurística clásica SPFF: siempre ruta más corta + primer bloque libre. Rápida pero ciega: insiste aunque la ruta
  esté llena.
- DRL aprende políticas adaptativas (DeepRMSA, Chen 2019: −20.3 % vs KSP-FF), PERO es muy sensible a
  hiperparámetros, fijados a mano (Doherty et al. 2025: heurísticas bien configuradas igualan a agentes publicados).
- Dilema que aprende el agente: desviarse a una ruta larga gasta más slots (modulación peor: 100G a 2090 km = QPSK
  4 slots; a 4860 km = BPSK 8 slots), pero puede evitar un bloqueo.

Tabla modulación (slots por bitrate): BPSK 5520 km 1/4/8/32 · QPSK 2720 km 1/2/4/16 · 8QAM 1360 km 1/2/3/11 ·
16QAM 560 km 1/1/2/8 · 32QAM 240 km 1/1/2/7 · 64QAM 80 km 1/1/2/6.

## 2. Objetivos (Taller 4) — los 5 cumplidos
General: Optimizar con Optuna los hiperparámetros de un agente DRL para reducir la probabilidad de bloqueo en RMLSA
dinámico en EON, usando DREAM-ON GYM.
1. Agente base PPO (SB3, por defecto) + escenario (topología, tráfico, espectro) → ✔ etapa 1 (03-11).
2. Espacio de búsqueda lr, gamma, batch size → ✔.
3. Optimización con Optuna, objetivo = bloqueo → ✔ 300 trials.
4. Comparar optimizado vs base vs SPFF en distintas cargas → ✔ 10 cargas × 10 semillas.
5. Identificar hiperparámetros más influyentes → ✔ PED-ANOVA (gamma 0.64).
(Resultados esperados del T4: menor bloqueo que defecto ✔, comparable o superior a SPFF ✔ superior, identificar
influyentes ✔, poda reduce tiempo ✔ ~5.6 h.)

## 3. Metodología
Pipeline: Escenario (DREAM-ON GYM) → Agente PPO → Optuna (300 trials) → Evaluación (semillas nuevas) → Análisis de red.

Escenario: NSFNet 14 nodos, 21 enlaces bidireccionales (42 dirigidos), 250–2840 km, 100 slots/enlace, k=3 rutas por
par, bitrates 10/40/100/400 Gbps uniformes (1000G excluido: 80 de 100 slots con BPSK, se bloquea siempre),
6 modulaciones (la más eficiente que alcance), First-Fit. Entrenamiento a 100 Erlang (SPFF ~10 % bloqueo, hay margen).
Warm-up 3000 solicitudes con SPFF.

MDP (RMLSAEnv propio, Gymnasium sobre simulador simNetPy de DREAM-ON GYM):
- Acción: ruta 0, 1 o 2. Modulación y espectro = mismas reglas que SPFF → comparación aísla el valor de la ruta.
- Observación 47 valores: origen one-hot (14) + destino one-hot (14) + bitrate one-hot (4) + por ruta (×3):
  saltos, slots requeridos, slots libres, posición primer bloque que cabe, si cabe.
- Recompensa +1 asignada / −1 bloqueada. Episodio 1000 solicitudes, la red no se vacía.
- PPO SB3: actor-crítico, rollout 2048, clip; MLP 2×64 actor + 2×64 crítico (~15 000 parámetros). CPU.
- Por qué entorno propio: RlOnEnv de DREAM-ON GYM 0.0.9 roto (TypeError al crear eventos, observación constante).
  Además: declara TensorFlow/CUDA/MPI sin usarlos (instalamos sin ellos), falta bitRates.schema.json (instalación
  desde fuente).

Optuna:
- Espacio: learning_rate 1e-5–1e-2 (log), gamma 0.8–0.9999 (lineal), batch_size {32,64,128,256,512} (categórica).
  Defaults SB3: 3e-4, 0.99, 64 (incluidos en el rango).
- TPESampler(seed=0): 10 primeros aleatorios, luego separa buenos/malos y propone "parecido a lo bueno".
  De 291 trials posteriores, 283 con lr < 2e-4 y 289 con gamma < 0.9 (se concentró en la zona buena).
- MedianPruner(n_startup_trials=5, n_warmup_steps=1): 200 000 pasos en 5 etapas de 40 000; tras cada etapa mide
  bloqueo; si peor que la mediana → poda.
- Objetivo: bloqueo con 10 000 solicitudes, promedio de 3 semillas de tráfico (1000–1002).
- SQLite (results/optuna.db): reanudable.
- Semillas separadas: 0 entrenamiento; 1000–1002 Optuna; 2000–2009 prueba final (nadie las vio); entrenamientos
  finales con semillas 0, 1, 2.
- Evaluación final: 3 entrenamientos × 300 000 pasos por configuración; 10 cargas (25–250 Erl) × 10 semillas ×
  10 000 solicitudes; media ± IC 95 %; mejora pareada por semilla.
- Evolución: v1 30 trials (0.0763, γ 0.987, 1 semilla) → v2 40 (0.0726, γ 0.830) → v3 100 (0.0756, γ 0.836, 3
  semillas) → v4 300 (0.0753). Lección: con 1 semilla las conclusiones se invertían (v1 decía gamma alto).

## 4. Demostración
Video: evidencias/2026-11-20_300trials/23_maqueta_150erl.mp4 (≈33 s, 6000 solicitudes, 150 Erl, semilla 2000,
modelo optuna_s0). Cuadro: 24_maqueta_cuadro.png. Izquierda SPFF, derecha PPO Optuna, misma secuencia. Gris =
ocupación del enlace, rojo = bloqueada, celeste = aceptada por ruta alternativa. Abajo: bloqueo acumulado.
A ~5430 solicitudes: SPFF 771 (14.20 %) vs PPO 711 (13.09 %), 212 desvíos, 60 bloqueos menos (−7.8 %).
Qué decir: "mismo tráfico, la única diferencia es la ruta elegida; fíjense en el celeste: conexiones que SPFF pierde
y el agente salva yendo por otro camino". Ilustración; las cifras oficiales son las de la evaluación.
Modo interactivo (respaldo): python scripts\maqueta_red.py (sliders carga 25–250, velocidad, agente). Recomendado
llevar el video (no puede fallar en vivo).

## 5. Resultados
Optuna 300 trials: 90 completos, 210 podados (+1 FAIL). 487 min de cómputo (8.1 h); sin poda ~13.7 h → ahorro ~5.6 h.
Trial promedio completo 2.7 min. Mejor: trial #210, 0.0753, gamma 0.800, lr 1.81e-5, batch 128.
Top 5: #210 0.0753 (γ0.800, lr1.8e-5, 128) · #148 0.0755 (0.827, 5.8e-5, 512) · #290 0.0755 (0.829, 1.1e-4, 512) ·
#30 0.0756 (0.836, 2.2e-5, 64) · #208 0.0757 (0.804, 1.9e-5, 128). → misma receta: gamma 0.80–0.84, lr bajo.
Importancia PED-ANOVA: gamma 0.640, batch 0.202, lr 0.158 (con 100 trials: lr 0.72, gamma 0.19). Cambia el orden
porque se mide dentro de lo explorado (TPE concentró la búsqueda). Robusto: DÓNDE están los buenos.
100→300 trials mejoró solo 0.0003 (< ruido entre semillas) → óptimo robusto.

Bloqueo (media ± IC95, semillas 2000–2009):
| Erl | SPFF | PPO defecto | PPO Optuna | Opt vs SPFF | Opt vs def | def vs SPFF |
| 25 | 0.0107±0.0009 | 0.0038±0.0005 | 0.0020±0.0004 | 81.0 % | 46.7 % | 64.6 % |
| 50 | 0.0398±0.0017 | 0.0279±0.0009 | 0.0204±0.0013 | 48.6 % | 26.8 % | 29.7 % |
| 75 | 0.0718±0.0017 | 0.0606±0.0016 | 0.0513±0.0017 | 28.5 % | 15.4 % | 15.6 % |
| 100 | 0.0985±0.0024 | 0.0891±0.0019 | 0.0798±0.0018 | 18.9 % | 10.4 % | 9.5 % |
| 125 | 0.1202±0.0022 | 0.1126±0.0018 | 0.1038±0.0022 | 13.6 % | 7.8 % | 6.2 % |
| 150 | 0.1416±0.0032 | 0.1328±0.0018 | 0.1230±0.0021 | 13.0 % | 7.4 % | 6.1 % |
| 175 | 0.1591±0.0031 | 0.1498±0.0027 | 0.1393±0.0024 | 12.4 % | 7.0 % | 5.8 % |
| 200 | 0.1757±0.0026 | 0.1662±0.0026 | 0.1567±0.0020 | 10.8 % | 5.7 % | 5.4 % |
| 225 | 0.1912±0.0025 | 0.1812±0.0023 | 0.1696±0.0024 | 11.3 % | 6.4 % | 5.2 % |
| 250 | 0.2057±0.0030 | 0.1955±0.0021 | 0.1821±0.0023 | 11.5 % | 6.9 % | 5.0 % |
IC de mejora Optuna vs SPFF: ±3.3 (25) … ±1.8 (100) … ±1.1 (250). Optuna vs def a 100: 10.4 ± 1.2 %.
- Gana en 100/100 casos a SPFF y 100/100 a PPO defecto. Entrenó solo a 100 Erl → generaliza a todas.
- Mejora relativa baja con la carga (81 % → 11 %): red saturada, ninguna ruta tiene espacio.
- Consistencia a 100 Erl: Optuna 0.0785–0.0811 (rango 0.0026); defecto 0.0850–0.0935 (0.0085) → ~3× más dispersión.
- El peor optimizado < mejor defecto en 89 de 100 casos.
- Curvas: Optuna baja de 10 % a ~40 000 pasos (39k–40k, las 3 iguales); defecto 37k–73k, irregular. Ambas planas
  antes de 150 000; se entrenó 300 000.
- Optuna (0.0753) vs evaluación (0.0798 a 100 Erl): la de Optuna es optimista (mejor de 300, mismas semillas); la
  oficial es la de semillas nuevas.

Por qué gana (análisis de red, 100 Erl):
- Desvíos de la ruta más corta: PPO Optuna 1.2 % (25 Erl) → 9.6 % (100) → 17.4 % (250). PPO defecto ~30–33 %
  siempre. SPFF 0 %. → aprendió CUÁNDO desviarse (nadie se lo programó).
- Bloqueo por bitrate (SPFF / def / Optuna): 10G 0.0003/0.0017/0.0000 · 40G 0.0065/0.0008/0.0006 ·
  100G 0.0356/0.0085/0.0077 · 400G 0.3515/0.3456/0.3112. Casi todo el bloqueo es de 400G (hasta 32 slots contiguos).
- Por distancia del par: <1000 km 0.0051/0.0030/0.0036 · 1–2k 0.0473/0.0467/0.0445 · 2–3k 0.1047/0.0881/0.0766 ·
  >3000 km 0.1919/0.1782/0.1592.
- Ocupación media del espectro: SPFF 27.3 %, defecto 31.8 %, Optuna 31.3 % → acepta más conexiones y reparte la
  carga (mapa: SPFF satura el corredor central 1–3–4–6–7 / 3–10–11; agente usa el sur 1–2–5–13).
- Enlace Seattle(0)–Champaign(7) pasa visualmente bajo Lincoln(6): es un solo enlace de 2840 km.

Figuras disponibles (evidencias/2026-11-20_300trials/): 04 historial, 05 importancia, 06 slice, 07 coord.
paralelas, 08 valores intermedios, 09 gamma vs lr, 11 bloqueo vs carga (log, IC), 12 mejora vs carga,
13 curvas aprendizaje, 14 uso rutas, 15 bloqueo por bitrate, 16 por distancia, 17 mapa ocupación, 23 video,
24 cuadro. informe/fig/: espectro.png, gamma.png, nsfnet.png, pipeline.png, maqueta_corto.png.

## 6. Autocrítica y trabajo futuro
Autocrítica:
- Mejor gamma 0.800 = borde inferior del rango → el rango quedó corto; abrirlo hacia abajo.
- Una topología (NSFNet), una carga de entrenamiento (100 Erl).
- Agente solo elige ruta (k=3); modulación y espectro fijos (First-Fit).
- Solo 3 hiperparámetros; resto de PPO por defecto (n_steps, clip, ent_coef, arquitectura).
- Una sola heurística de referencia (SPFF; se quitó KSP-FF para ajustarse a la propuesta). Todo simulación.
- Estudios v1/v2 con 1 semilla daban conclusiones engañosas → corregido desde v3 (costó tiempo).
- No usamos el entorno oficial de DREAM-ON GYM (roto) → entorno propio.
- La importancia de hiperparámetros no es absoluta (depende de lo explorado).
- 400G sigue bloqueándose ~31 %: el agente no resuelve la fragmentación.
Trabajo futuro:
- Abrir rango de gamma (<0.8) y optimizar más hiperparámetros.
- Otras topologías (USNet, COST239…), entrenar en varias cargas.
- Agente que elija también modulación/espectro; más rutas (k>3); otras heurísticas (KSP-FF, etc.).
- QKD: canales clásicos + cuánticos (propuesto en T4).
- Orquestador multi-agente con LLM que use al agente como herramienta (solo mencionar).

## 7. Conclusiones (mensaje final)
- Optuna encontró automáticamente una configuración que reduce el bloqueo 11–81 % vs SPFF y 6–47 % vs PPO defecto,
  en todas las cargas y semillas, y hace el entrenamiento más estable (3× menos dispersión).
- Hiperparámetro clave: gamma bajo (~0.8): el efecto de una decisión se nota en las próximas ~5 solicitudes, no 100.
- El valor del DRL bien ajustado: aprender cuándo desviarse.
- Todo en un notebook sin GPU, ~10 h de cómputo en total.

## 8. Preguntas probables (respuestas)
- ¿Optuna entrena al agente? No; propone configs y poda; PPO entrena dentro de cada trial.
- ¿Qué decide el agente? Solo la ruta (1 de 3); mod. y espectro iguales a SPFF → aísla el valor de la ruta.
- ¿Por qué gana a SPFF con las mismas reglas? SPFF insiste en la corta aunque esté llena; el agente ve espacio libre
  por ruta y si cabe.
- ¿Por qué gana a PPO defecto? Defecto se desvía ~30 % siempre, gasta espectro; Optuna selectivo según carga.
- ¿Por qué gamma tan bajo? 0.99 es genérico (juegos/robótica). Aquí solicitudes aleatorias independientes; horizonte
  1/(1−γ): 0.99 → ~100 solicitudes; 0.80 → ~5. Mirar lejos mete ruido.
- ¿Por qué 300 trials? Profesora sugirió; se extendió v3 (100) a 300; mejoró 0.0003, misma zona.
- ¿Trampa con las semillas? No: Optuna 1000–1002, prueba 2000–2009, agentes nuevos con 3 semillas.
- ¿No es azar? IC95 de mejora lejos de cero, 100/100 casos, 3 entrenamientos, estudios distintos convergen.
- ¿Poda? Abandona trial peor que la mediana en esa etapa; 210/300 podados; ~5.6 h ahorradas.
- ¿Entorno oficial? Roto en 0.0.9; usamos el simulador del framework + capa Gymnasium propia.
- ¿Entrenar más? Curvas planas antes de 150k; entrenamos 300k (1M descartado).
- ¿Por qué el gamma en el borde? Limitación; trabajo futuro abrir rango.
- ¿Por qué ocupa más espectro que SPFF? Acepta más conexiones y usa rutas más largas cuando conviene.
- ¿GPU? No ayuda: cuello de botella es el simulador en Python paso a paso; red de 15k parámetros.
- ¿Por qué rápido? Formulación compacta (3 acciones, 47 entradas vs cientos en DeepRMSA), recompensa inmediata,
  poda + TPE (grilla 10×10×5 = 500 entrenamientos), bloqueos 1–20 % se miden bien con 10 000 solicitudes.
- ¿Por qué 1000G excluido? 80 de 100 slots con BPSK → se bloquea siempre y tapa el efecto del agente.
- ¿Por qué 100 Erlang? SPFF ~10 % bloqueo: margen para mejorar sin red vacía ni saturada.

## Referencias IEEE (T4 + informe de estudio)
[1] O. Gerstel, M. Jinno, A. Lord, S. J. B. Yoo, "Elastic optical networking: A new dawn for the optical layer?" IEEE
Commun. Mag., vol. 50, no. 2, pp. s12–s20, Feb. 2012.
[2] K. Christodoulopoulos, I. Tomkos, E. A. Varvarigos, "Elastic bandwidth allocation in flexible OFDM-based optical
networks," J. Lightw. Technol., vol. 29, no. 9, pp. 1354–1366, May 2011.
[3] X. Chen et al., "DeepRMSA...," J. Lightw. Technol., vol. 37, no. 16, pp. 4155–4163, Aug. 2019.
[4] N. Jara, H. Pempelfort, E. Viera, J. P. Sanchez, G. España, D. Bórquez-Paredes, "DREAM-ON GYM...," SIMULTECH 2024,
pp. 215–222.
[5] M. Doherty, R. Matzner, R. Sadeghi, P. Bayvel, A. Beghelli, "Reinforcement learning for dynamic resource
allocation in optical networks: Hype or hope?" J. Opt. Commun. Netw., vol. 17, p. D1, 2025.
[6] T. Akiba et al., "Optuna...," KDD 2019, pp. 2623–2631.
[7] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[8] A. Raffin et al., "Stable-Baselines3...," JMLR, vol. 22, 2021.
[9] J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, "Algorithms for hyper-parameter optimization," NeurIPS 2011.

## Estado del informe IEEE (Overleaf) al 24-11-2026
- Nicolás arma el .tex en Overleaf (base = Taller 5, ex Taller 4). Claude entrega cada sección como .txt abierto en
  el Bloc de notas; él copia/pega y avisa; se borra y se abre el siguiente.
- Estructura = la de la PAUTA del Taller Final (no la del Taller 5):
  I Descripción del problema · II Objetivos · III Metodología · IV Demostración · V Resultados ·
  VI Autocrítica y trabajo futuro · VII Conclusiones. (Se eliminaron "Estado del arte" y "Resultados esperados";
  lo importante del estado del arte quedó en el 3er párrafo de la Sección I.)
- LISTO y pegado: título, autores, resumen, keywords, I, II, III (texto completo en
  `informe_avance_hasta_metodologia.tex.txt`, desde \begin{abstract} hasta el fin de Metodología).
- Bib: claves existentes gerstel, christodoulopoulos, deeprmsa, dreamongym, doherty2025hype, optuna; agregadas
  sb3, ppo, tpe (las 3 entradas están al final del .tex.txt, van en references.bib).
- Tablas ya usadas: tab:escenario, tab:espacio.
- PENDIENTE: IV Demostración, V Resultados, VI Autocrítica y trabajo futuro, VII Conclusiones.

## Imágenes de la presentación (carpeta del escritorio "Imagenes_Presentacion_TEL317")
Orden de 12 páginas: 1 Portada · 2 Problema · 3 Por qué DRL + Optuna · 4 Objetivos · 5 Metodología escenario ·
6 Metodología Optuna · 7 Demostración · 8 Resultados Optuna · 9 Resultados bloqueo · 10 Por qué gana ·
11 Autocrítica y trabajo futuro · 12 Conclusiones.
| Pág. | Archivo (en el repo) |
|---|---|
| 2 | informe/fig/espectro.png |
| 5 | informe/fig/nsfnet.png |
| 6 | informe/fig/gamma.png |
| 7 | evidencias/2026-11-20_300trials/23_maqueta_150erl.mp4 (respaldo: 24_maqueta_cuadro.png) |
| 8 | evidencias/2026-11-20_300trials/09_optuna_gamma_vs_learning_rate.png (importancia gamma 0,64 como dato, no imagen) |
| 9 | evidencias/2026-11-20_300trials/11_bloqueo_vs_carga.png |
| 10 | evidencias/2026-11-20_300trials/14_uso_rutas.png |
Respaldo para preguntas: 04 historial, 13 curvas, 16 distancia, 15 bitrate, 17 mapa, 12 mejora, 05 importancia,
01/02/03 capturas de terminal. NO usar informe/fig/pipeline.png (dice 100 trials, desactualizado).

Pendientes de la presentación: nombre del equipo (portada), confirmar demo con video (recomendado) vs ventana interactiva.
