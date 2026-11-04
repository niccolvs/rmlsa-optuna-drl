# Escenario de simulación (objetivo específico 1)

## Red y tráfico

| Parámetro             | Valor                                                        |
|-----------------------|--------------------------------------------------------------|
| Topología             | NSFNet: 14 nodos, 21 enlaces bidireccionales (250–2840 km)   |
| Espectro              | 100 slots por enlace                                         |
| Rutas candidatas      | k = 3 por par origen-destino (de menos a más saltos)         |
| Llegadas              | Poisson, tasa λ = carga [Erlang] (μ = 1)                     |
| Tiempo de retención   | Exponencial, media 1/μ = 1                                   |
| Bitrates              | 10, 40, 100, 400 Gbps, uniformes                             |
| Modulaciones          | BPSK, QPSK, 8/16/32/64-QAM: la más eficiente que alcance     |
| Asignación espectral  | First-Fit (contigüidad y continuidad)                        |

1000 Gbps se excluyó: con BPSK ocupa 80 de 100 slots y se bloquea casi siempre, sin importar la
ruta, lo que tapa el efecto del agente.

## Agente (MDP)

- **Acción**: cuál de las 3 rutas usar. Modulación y espectro los resuelve la regla de arriba.
- **Observación** (48 valores): origen y destino (one-hot), bitrate (one-hot) y, por ruta:
  saltos, slots requeridos, slots libres, posición del primer bloque que cabe, y si cabe o no.
- **Recompensa**: +1 si la conexión se asigna, −1 si se bloquea.
- **Episodio**: 1000 solicitudes; la red no se vacía entre episodios.
- **Warm-up**: 3000 solicitudes con SPFF antes de que actúe el agente.

## Bloqueo de referencia (3000 solicitudes, semilla 0)

| Carga [Erl] | SPFF   |
|-------------|--------|
| 50          | 0.0283 |
| 100         | 0.0843 |
| 200         | 0.1640 |
| 300         | 0.2120 |

Carga principal propuesta para entrenar: **100 Erlang** (SPFF bloquea ~8 %, hay margen para mejorar).

## Por qué no se usa `RlOnEnv` de DREAM-ON GYM

Su `RlOnEnv` usa `simulator_finite.py`, que en la versión 0.0.9 falla al crear eventos
(`TypeError: Event.__init__() takes 4 positional arguments but 7 were given`), y entrega una
observación constante. `RMLSAEnv` usa el `Simulator` estándar de `dreamongymv2.simNetPy`, con las
mismas funciones de asignación.
