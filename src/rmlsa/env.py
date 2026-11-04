"""Entorno Gymnasium para RMLSA dinámico sobre el simulador de DREAM-ON GYM.

Se usa directamente ``dreamongymv2.simNetPy.Simulator`` en vez de su
``RlOnEnv``: este último depende de ``simulator_finite``, que está roto en la
versión 0.0.9 (TypeError al crear eventos), y además entrega una observación
constante, con la que el agente no puede aprender nada.

Cada paso = una solicitud de conexión que llega a la red.
    Acción: índice de la ruta candidata (0..K-1). La modulación es la más
        eficiente que alcanza la distancia y el espectro se asigna con
        First-Fit, así que el agente decide la parte de "R" de RMLSA.
    Recompensa: +1 si la conexión se asigna, -1 si se bloquea.
    Episodio: ``episode_length`` solicitudes. El tráfico continúa entre
        episodios (la red no se vacía al hacer reset).
"""

from pathlib import Path

import gymnasium
import numpy as np
from dreamongymv2.simNetPy import Controller, Simulator

from rmlsa.heuristics import (
    first_fit,
    first_fit_start,
    modulation_slots,
    path_length,
    used_slots,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOPOLOGY = DATA_DIR / "topologies" / "NSFNet.json"
ROUTES = DATA_DIR / "topologies" / "NSFNet_routes.json"
BITRATES = DATA_DIR / "bitrates" / "bitrates_nsfnet.json"

K_PATHS = 3  # NSFNet_routes.json trae 3 rutas candidatas por par origen-destino
FEATURES_PER_PATH = 5


class RMLSAEnv(gymnasium.Env):
    """RMLSA dinámico: llegadas Poisson, tiempos de retención exponenciales.

    Args:
        load: carga de tráfico en Erlang (= lambda / mu, con mu = 1).
        seed: semilla base del tráfico (llegadas, salidas, pares y bitrates).
        episode_length: solicitudes por episodio.
        warmup: solicitudes atendidas con SPFF antes de que actúe el agente,
            para que la red parta "cargada" y no vacía.
        allocator: función de asignación; por defecto, ruta del agente + FF.
    """

    def __init__(self, load=300.0, seed=0, episode_length=1000, warmup=3000, allocator=first_fit):
        super().__init__()
        self.load = load
        self.traffic_seed = seed
        self.episode_length = episode_length
        self.warmup = warmup
        self.allocator = allocator
        self.sim = None

        self.action_space = gymnasium.spaces.Discrete(K_PATHS)
        # nodo origen (one-hot) + destino (one-hot) + bitrate (one-hot) + features por ruta
        self._n_nodes = 14
        self._n_bitrates = 4
        size = 2 * self._n_nodes + self._n_bitrates + K_PATHS * FEATURES_PER_PATH
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(size,), dtype=np.float32)

    # --- simulador -----------------------------------------------------------

    def _build_simulator(self):
        sim = Simulator(str(TOPOLOGY), str(ROUTES), str(BITRATES))
        sim.setLambda(self.load)
        sim.setMu(1.0)
        s = self.traffic_seed
        sim.setSeedArrive(1000 + s)
        sim.setSeedDeparture(2000 + s)
        sim.setSeedSrc(3000 + s)
        sim.setSeedDst(4000 + s)
        sim.setSeedBitRate(5000 + s)
        sim.setAllocator(self.allocator)
        sim.setGoalConnections(self.warmup)
        sim.init()
        sim.run()  # warm-up sin agente (action=None -> ruta más corta)
        sim.forwardDepartures()
        sim.createEventConnection()  # primera solicitud que verá el agente
        return sim

    @property
    def _request(self):
        ev = self.sim.connectionEvent
        return ev.source, ev.destination, ev.bitRate

    # --- observación ---------------------------------------------------------

    def _observation(self):
        src, dst, br = self._request
        network = self.sim.controller.network
        bitrate = self.sim.bitRatesDefault[br]
        total_slots = network.getLink(0).getSlots()

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[src] = 1.0
        obs[self._n_nodes + dst] = 1.0
        obs[2 * self._n_nodes + br] = 1.0

        base = 2 * self._n_nodes + self._n_bitrates
        for k, route in enumerate(self.sim.controller.path[src][dst][:K_PATHS]):
            used = used_slots(route, network)
            needed = modulation_slots(bitrate, path_length(route))
            start = None if needed is None else first_fit_start(used, needed)
            f = base + k * FEATURES_PER_PATH
            obs[f + 0] = len(route) / 6.0                            # saltos
            obs[f + 1] = -1.0 if needed is None else needed / 32.0   # slots requeridos
            obs[f + 2] = used.count(False) / total_slots             # slots libres
            obs[f + 3] = -1.0 if start is None else start / total_slots  # posición FF
            obs[f + 4] = 0.0 if start is None else 1.0               # ¿cabe?
        return obs

    # --- API Gymnasium -------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sim is None:
            self.sim = self._build_simulator()
        self._steps = 0
        self._blocked = 0
        return self._observation(), {}

    def step(self, action):
        self.sim.step(int(action))
        allocated = self.sim.lastConnectionIsAllocated().name == Controller.Status.Allocated.name
        self._steps += 1
        self._blocked += 0 if allocated else 1

        self.sim.forwardDepartures()
        self.sim.createEventConnection()

        truncated = self._steps >= self.episode_length
        info = {"allocated": allocated}
        if truncated:
            info["blocking_probability"] = self._blocked / self._steps
        return self._observation(), (1.0 if allocated else -1.0), False, truncated, info


def run_heuristic(allocator, load, seed, n_requests):
    """Probabilidad de bloqueo de una heurística sobre n_requests solicitudes."""
    env = RMLSAEnv(load=load, seed=seed, episode_length=n_requests, allocator=allocator)
    env.reset()
    info = {}
    while "blocking_probability" not in info:
        _, _, _, _, info = env.step(0)
    return info["blocking_probability"]


def evaluate_model(model, load, seed, n_requests):
    """Probabilidad de bloqueo de un agente SB3 (política determinista)."""
    env = RMLSAEnv(load=load, seed=seed, episode_length=n_requests)
    obs, info = env.reset()
    while "blocking_probability" not in info:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
    return info["blocking_probability"]
