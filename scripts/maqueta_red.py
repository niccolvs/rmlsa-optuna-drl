"""Maqueta de NSFNet operando en vivo: SPFF y un agente PPO atienden la MISMA secuencia de solicitudes.

Cada enlace se colorea según su ocupación de espectro. En cada cuadro se dibujan las solicitudes
bloqueadas (rojo) y las que el agente aceptó desviándolas por una ruta alternativa (celeste).
Abajo, la probabilidad de bloqueo acumulada de ambos.

Modo interactivo (ventana con controles: carga, velocidad, agente, pausa, reiniciar):
    python scripts/maqueta_red.py

Modo video (MP4 para la presentación):
    python scripts/maqueta_red.py --video results/maqueta_150erl.mp4 --carga 150
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models" / "final"
COLORS = {"SPFF": "#1f77b4", "PPO Optuna": "#2ca02c", "PPO defecto": "#ff7f0e"}
# Posiciones aproximadas de los nodos de NSFNet (las mismas de analisis_red.py)
POS = {0: (1, 9), 1: (0.4, 6.2), 2: (1.3, 3.4), 3: (3.2, 6.6), 4: (4.8, 5.6), 5: (6.2, 2.4), 6: (6.2, 6.6),
       7: (7.7, 7.3), 8: (10.0, 6.4), 9: (9.2, 2.8), 10: (8.9, 8.7), 11: (11.4, 8.3), 12: (10.9, 7.4), 13: (12.2, 5.0)}
BLOCKED, DETOUR = "#d00000", "#00b4d8"


class Agente:
    """Un simulador con su política. Ambos agentes usan la misma semilla de tráfico."""

    def __init__(self, name, load, seed, model=None):
        from rmlsa.env import RMLSAEnv
        from rmlsa.heuristics import first_fit, spff

        self.name, self.model = name, model
        with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
            self.env = RMLSAEnv(load=load, seed=seed, episode_length=10**9,
                                allocator=first_fit if model else spff)
            self.obs, _ = self.env.reset()
        self.network = self.env.sim.controller.network
        self.n_links = self.network.getNumberOfLinks()
        self.slots = self.network.getLink(0).getSlots()
        self.n = self.blocked = self.detours = 0

    def step(self):
        """Atiende una solicitud. Retorna (ruta usada, asignada, k)."""
        src, dst, _ = self.env._request
        routes = self.env.sim.controller.path[src][dst]
        action = 0 if self.model is None else int(self.model.predict(self.obs, deterministic=True)[0])
        k = min(action, len(routes) - 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.obs, _, _, _, info = self.env.step(action)
        self.n += 1
        self.blocked += 0 if info["allocated"] else 1
        self.detours += 1 if (info["allocated"] and k > 0) else 0
        return routes[k], info["allocated"], k

    def occupancy(self):
        out = {}
        for i in range(self.n_links):
            link = self.network.getLink(i)
            out[i] = sum(link.getSlot(s) for s in range(self.slots)) / self.slots
        return out

    @property
    def blocking(self):
        return self.blocked / self.n if self.n else 0.0


def load_models(run):
    from stable_baselines3 import PPO

    return {name: PPO.load(MODELS / run / f"{slug}_s0.zip") for name, slug in
            [("PPO Optuna", "optuna"), ("PPO defecto", "defecto")]}


class Maqueta:
    def __init__(self, fig, models, load, seed, per_frame, right="PPO Optuna", curve_box=(0.07, 0.08, 0.86, 0.22)):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap

        self.models, self.seed, self.per_frame = models, seed, per_frame
        self.fig = fig
        topo = json.load(open(ROOT / "data" / "topologies" / "NSFNet.json"))
        # un segmento por par de nodos (los dos sentidos del enlace se promedian)
        self.pairs = sorted({(min(l["src"], l["dst"]), max(l["src"], l["dst"])) for l in topo["links"]})
        self.link_pair = {l["id"]: self.pairs.index((min(l["src"], l["dst"]), max(l["src"], l["dst"])))
                          for l in topo["links"]}
        segs = [[POS[a], POS[b]] for a, b in self.pairs]

        # ocupación en grises (más oscuro = más lleno) para que resalten los eventos en rojo y celeste
        cmap = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.15, 0.85, 256)))
        self.axes, self.base, self.events = [], [], []
        for x0 in (0.02, 0.48):
            ax = fig.add_axes([x0, 0.37, 0.43, 0.50])
            base = LineCollection(segs, cmap=cmap, linewidths=8, zorder=1)
            base.set_array(np.zeros(len(segs)))
            base.set_clim(0, 0.6)
            ax.add_collection(base)
            ev = LineCollection([], linewidths=3.5, zorder=2, capstyle="round")
            ax.add_collection(ev)
            for n, (x, y) in POS.items():
                ax.scatter(x, y, s=300, color="#1d3557", zorder=3)
                ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=8, zorder=4)
            ax.set_xlim(-0.3, 12.9)
            ax.set_ylim(1.7, 9.7)
            ax.axis("off")
            self.axes.append(ax)
            self.base.append(base)
            self.events.append(ev)
        cax = fig.add_axes([0.935, 0.42, 0.012, 0.40])
        fig.colorbar(self.base[0], cax=cax, label="Fracción de slots ocupados")
        fig.text(0.5, 0.345, "rojo: solicitud bloqueada      celeste: aceptada por una ruta alternativa",
                 ha="center", fontsize=9, color="#444")

        self.ax_curve = fig.add_axes(curve_box)
        self.curve_lines = {}
        self.suptitle = fig.suptitle("", fontsize=13, fontweight="bold")
        self.restart(load, right)

    def restart(self, load, right):
        self.load, self.right = load, right
        self.agents = [Agente("SPFF", load, self.seed), Agente(right, load, self.seed, self.models[right])]
        self.history = [[], [], []]  # solicitudes, bloqueo SPFF, bloqueo derecha
        ax = self.ax_curve
        ax.clear()
        for i, a in enumerate(self.agents):
            (self.curve_lines[i],) = ax.plot([], [], color=COLORS[a.name], lw=2, label=a.name)
        ax.set_xlabel("Solicitudes atendidas")
        ax.set_ylabel("Bloqueo acumulado [%]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        self.suptitle.set_text(f"NSFNet en vivo a {load:.0f} Erlang: misma secuencia de solicitudes para ambos")

    def update(self, _=None):
        for agent, ax, base, ev in zip(self.agents, self.axes, self.base, self.events):
            segs, cols = [], []
            for _ in range(self.per_frame):
                route, ok, k = agent.step()
                if not ok or k > 0:
                    segs += [[POS[l.src], POS[l.dst]] for l in route]
                    cols += [BLOCKED if not ok else DETOUR] * len(route)
            ev.set_segments(segs)
            ev.set_color(cols)
            occ = agent.occupancy()
            vals = np.zeros(len(self.pairs))
            cnt = np.zeros(len(self.pairs))
            for lid, o in occ.items():
                vals[self.link_pair[lid]] += o
                cnt[self.link_pair[lid]] += 1
            base.set_array(vals / np.maximum(cnt, 1))
            extra = f"   desvíos: {agent.detours}" if agent.model else ""
            ax.set_title(f"{agent.name}\nbloqueadas: {agent.blocked} de {agent.n}  "
                         f"({agent.blocking * 100:.2f} %){extra}", fontsize=11, color=COLORS[agent.name])

        spff, ppo = self.agents
        self.history[0].append(spff.n)
        self.history[1].append(spff.blocking * 100)
        self.history[2].append(ppo.blocking * 100)
        for i in range(2):
            self.curve_lines[i].set_data(self.history[0], self.history[i + 1])
        self.ax_curve.set_xlim(0, max(1000, spff.n))
        top = max(max(self.history[1]), max(self.history[2]), 1)
        self.ax_curve.set_ylim(0, top * 1.25)
        diff = spff.blocked - ppo.blocked
        rel = diff / spff.blocked * 100 if spff.blocked else 0
        self.ax_curve.set_title(f"{ppo.name} bloqueó {diff} solicitudes menos que SPFF ({rel:.1f} % menos)"
                              if diff >= 0 else f"{ppo.name} bloqueó {-diff} solicitudes más que SPFF",
                                fontsize=11, fontweight="bold", color=COLORS[ppo.name])
        return []


def interactive(args, models):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, RadioButtons, Slider

    fig = plt.figure(figsize=(13, 8.3))
    fig.canvas.manager.set_window_title("Maqueta NSFNet: SPFF vs PPO")
    m = Maqueta(fig, models, args.carga, args.semilla, args.por_cuadro, curve_box=(0.07, 0.16, 0.86, 0.15))
    state = {"paused": False, "restart": False}

    s_load = Slider(fig.add_axes([0.14, 0.065, 0.30, 0.025]), "Carga [Erlang]", 25, 250,
                    valinit=args.carga, valstep=25)
    s_speed = Slider(fig.add_axes([0.14, 0.025, 0.30, 0.025]), "Solicitudes/cuadro", 1, 100,
                     valinit=args.por_cuadro, valstep=1)
    radio = RadioButtons(fig.add_axes([0.52, 0.01, 0.14, 0.09]), ["PPO Optuna", "PPO defecto"])
    b_pause = Button(fig.add_axes([0.70, 0.035, 0.10, 0.05]), "Pausa")
    b_reset = Button(fig.add_axes([0.82, 0.035, 0.10, 0.05]), "Reiniciar")

    def ask_restart(_=None):
        state["restart"] = True  # se reinicia en el próximo cuadro, no dentro del callback

    def set_speed(v):
        m.per_frame = int(v)

    def toggle(_):
        state["paused"] = not state["paused"]
        b_pause.label.set_text("Continuar" if state["paused"] else "Pausa")

    s_load.on_changed(ask_restart)
    radio.on_clicked(ask_restart)
    s_speed.on_changed(set_speed)
    b_pause.on_clicked(toggle)
    b_reset.on_clicked(ask_restart)

    def frame(_):
        if state["restart"]:
            state["restart"] = False
            m.restart(s_load.val, radio.value_selected)
        if not state["paused"]:
            m.update()
        return []

    anim = FuncAnimation(fig, frame, interval=40, cache_frame_data=False)
    fig._anim = anim  # mantener la referencia viva
    plt.show()


def video(args, models):
    import imageio_ffmpeg
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    fig = plt.figure(figsize=(13, 7.6))
    m = Maqueta(fig, models, args.carga, args.semilla, args.por_cuadro, right=args.agente)
    frames = args.solicitudes // args.por_cuadro
    t0 = time.time()

    def progress(i, n):
        if i % 50 == 0:
            print(f"  cuadro {i}/{n} ({time.time() - t0:.0f} s)")

    anim = FuncAnimation(fig, m.update, frames=frames, cache_frame_data=False)
    out = Path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=110, progress_callback=progress)
    spff, ppo = m.agents
    print(f"Video en {out} ({frames} cuadros, {frames / args.fps:.0f} s). "
          f"Bloqueo SPFF {spff.blocking:.4f}, {ppo.name} {ppo.blocking:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carga", type=float, default=150, help="carga inicial en Erlang")
    parser.add_argument("--semilla", type=int, default=2000, help="semilla del tráfico")
    parser.add_argument("--por-cuadro", type=int, default=15, help="solicitudes atendidas por cuadro")
    parser.add_argument("--agente", default="PPO Optuna", choices=["PPO Optuna", "PPO defecto"],
                        help="agente del panel derecho (en el video)")
    parser.add_argument("--run", default="ppo_rmlsa_100erl_v4", help="carpeta de modelos en results/models/final")
    parser.add_argument("--video", help="si se indica, guarda un MP4 en esta ruta en vez de abrir la ventana")
    parser.add_argument("--solicitudes", type=int, default=6000, help="largo del video en solicitudes")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    if args.video:
        matplotlib.use("Agg")
    print("Cargando modelos...")
    if not (MODELS / args.run).exists():
        print(f"No existe {args.run}, uso ppo_rmlsa_100erl_v3")
        args.run = "ppo_rmlsa_100erl_v3"
    models = load_models(args.run)
    (video if args.video else interactive)(args, models)


if __name__ == "__main__":
    main()
