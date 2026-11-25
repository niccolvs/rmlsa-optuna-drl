"""Entorno Gymnasium para RMLSA sobre el simulador de DREAM-ON GYM."""

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

K_PATHS = 3  # 3 rutas por par
FEATURES_PER_PATH = 5


class RMLSAEnv(gymnasium.Env):
    def __init__(self, load=300.0, seed=0, episode_length=1000, warmup=3000, allocator=first_fit):
        super().__init__()
        self.load = load
        self.traffic_seed = seed
        self.episode_length = episode_length
        self.warmup = warmup
        self.allocator = allocator
        self.sim = None

        self.action_space = gymnasium.spaces.Discrete(K_PATHS)
        # origen + destino + bitrate + features por ruta
        self._n_nodes = 14
        self._n_bitrates = 4
        size = 2 * self._n_nodes + self._n_bitrates + K_PATHS * FEATURES_PER_PATH
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(size,), dtype=np.float32)

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
        sim.run()  # warm-up con SPFF
        sim.forwardDepartures()
        sim.createEventConnection()
        return sim

    @property
    def _request(self):
        ev = self.sim.connectionEvent
        return ev.source, ev.destination, ev.bitRate

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
    env = RMLSAEnv(load=load, seed=seed, episode_length=n_requests, allocator=allocator)
    env.reset()
    info = {}
    while "blocking_probability" not in info:
        _, _, _, _, info = env.step(0)
    return info["blocking_probability"]


def evaluate_model(model, load, seed, n_requests):
    env = RMLSAEnv(load=load, seed=seed, episode_length=n_requests)
    obs, info = env.reset()
    while "blocking_probability" not in info:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
    return info["blocking_probability"]


def evaluate_detailed(load, seed, n_requests, model=None, sample_every=100):
    """Retorna bloqueo, rutas, bitrate, distancia y ocupación de enlaces."""
    from rmlsa.heuristics import spff

    env = RMLSAEnv(load=load, seed=seed, episode_length=n_requests, allocator=first_fit if model else spff)
    obs, info = env.reset()
    network = env.sim.controller.network
    n_links = network.getNumberOfLinks()
    total_slots = network.getLink(0).getSlots()
    rutas, bitrate, distancia = {}, {}, {}
    occupancy = [0.0] * n_links
    samples = 0
    while "blocking_probability" not in info:
        if env._steps % sample_every == 0:
            for i in range(n_links):
                link = network.getLink(i)
                occupancy[i] += sum(link.getSlot(s) for s in range(total_slots)) / total_slots
            samples += 1
        src, dst, br = env._request
        routes = env.sim.controller.path[src][dst]
        action = 0 if model is None else int(model.predict(obs, deterministic=True)[0])
        k = min(action, len(routes) - 1)
        gbps = int(env.sim.bitRatesDefault[br].bitRate)
        km = int(path_length(routes[0]))
        obs, _, _, _, info = env.step(action)
        blocked = 0 if info["allocated"] else 1
        rutas[k] = rutas.get(k, 0) + 1
        n, b = bitrate.get(gbps, (0, 0)); bitrate[gbps] = (n + 1, b + blocked)
        n, b = distancia.get(km, (0, 0)); distancia[km] = (n + 1, b + blocked)
    return {"bloqueo": info["blocking_probability"], "rutas": rutas, "bitrate": bitrate, "distancia": distancia,
            "enlaces": {i: occupancy[i] / samples for i in range(n_links)}}
