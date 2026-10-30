"""Construcción del entorno RMLSA sobre DREAM-ON GYM.

El entorno de DREAM-ON GYM es un esqueleto: el usuario define la recompensa,
el estado y el algoritmo de asignación. Aquí se arma una versión base:
el agente elige la ruta (acción = índice entre las k rutas candidatas) y el
espectro se asigna con First-Fit usando la modulación más eficiente que
alcance la distancia de la ruta.
"""

from pathlib import Path

import gymnasium
from dreamongymv2.gym_basic import envs  # noqa: F401  (registra "rlonenv-v0")
from dreamongymv2.simNetPy import Controller

from rmlsa.heuristics import first_fit

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOPOLOGY = DATA_DIR / "topologies" / "NSFNet.json"
ROUTES = DATA_DIR / "topologies" / "NSFNet_routes.json"
BITRATES = DATA_DIR / "bitrates" / "bitrates_banda_C.json"

K_PATHS = 3  # NSFNet_routes.json trae 3 rutas candidatas por par origen-destino


def make_env(
    lam=1000.0,
    mu=100.0,
    seed=12345,
    warmup_connections=1000,
    allocator=first_fit,
):
    """Crea y deja listo un entorno RMLSA.

    Args:
        lam: tasa de llegadas (Poisson). La carga en Erlang es lam / mu.
        mu: tasa de salidas (tiempos de retención exponenciales).
        seed: semilla para llegadas, salidas, origen, destino y bitrate.
        warmup_connections: conexiones simuladas antes de que actúe el agente.
        allocator: función de asignación (src, dst, b, c, n, path, action).

    Returns:
        El entorno sin wrappers (``RlOnEnv``) con el simulador inicializado.
    """
    env = gymnasium.make(
        "rlonenv-v0",
        action_space=K_PATHS,
        start_training=warmup_connections,
    ).unwrapped

    def reward():
        status = env.getSimulator().lastConnectionIsAllocated()
        return -1 if status.name == Controller.Status.Not_Allocated.name else 1

    env.setRewardFunc(reward)
    env.initEnviroment(str(TOPOLOGY), str(ROUTES), str(BITRATES))

    sim = env.getSimulator()
    sim.setLambda(lam)
    sim.setMu(mu)
    sim.setSeedArrive(seed)
    sim.setSeedDeparture(seed + 1)
    sim.setSeedSrc(seed + 2)
    sim.setSeedDst(seed + 3)
    sim.setSeedBitRate(seed + 4)
    sim.setAllocator(allocator)
    return env


def blocking_probability(env):
    return env.getSimulator().getBlockingProbability()
