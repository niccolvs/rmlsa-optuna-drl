"""Figuras explicativas para el super informe (se guardan en informe/fig)."""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 10})

# 1. Topología NSFNet ----------------------------------------------------------
pos = {0: (1, 9), 1: (0.4, 6.2), 2: (1.3, 3.4), 3: (3.2, 6.6), 4: (4.8, 5.6), 5: (6.2, 2.4),
       6: (6.2, 6.6), 7: (7.7, 7.3), 8: (10.0, 6.4), 9: (9.2, 2.8), 10: (8.9, 8.7),
       11: (11.4, 8.3), 12: (10.9, 7.4), 13: (12.2, 5.0)}
topo = json.load(open(REPO / "data/topologies/NSFNet.json"))
edges = sorted({(min(l["src"], l["dst"]), max(l["src"], l["dst"]), l["length"]) for l in topo["links"]})
fig, ax = plt.subplots(figsize=(8, 4.6))
for a, b, km in edges:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.plot([x1, x2], [y1, y2], color="#7a8ba0", lw=1.6, zorder=1)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{km}", fontsize=7.5, color="#333",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=2)
for route, color in [([0, 1, 3], "#2a9d8f"), ([0, 7, 6, 4, 3], "#e76f51")]:
    xs, ys = zip(*[pos[n] for n in route])
    ax.plot(xs, ys, color=color, lw=4.5, alpha=0.45, zorder=0)
for n, (x, y) in pos.items():
    ax.scatter(x, y, s=420, color="#1d3557", zorder=3)
    ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
ax.plot([], [], color="#2a9d8f", lw=4.5, alpha=0.6, label="Ruta 1 de 0→3: 0-1-3 (2 saltos, 2090 km)")
ax.plot([], [], color="#e76f51", lw=4.5, alpha=0.6, label="Ruta 2 de 0→3: 0-7-6-4-3 (4 saltos, 4860 km)")
ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.set_title("NSFNet: 14 nodos, 21 enlaces bidireccionales (largo en km)", fontsize=11)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "nsfnet.png", dpi=200)
plt.close(fig)

# 2. Espectro: continuidad, contigüidad y First-Fit ------------------------------
fig, ax = plt.subplots(figsize=(8, 2.9))
rng = np.random.default_rng(3)
n_slots = 24
occ = {
    "Enlace 0→1": [0, 1, 2, 3, 8, 9, 14, 15, 16, 20],
    "Enlace 1→3": [0, 1, 4, 5, 6, 9, 17, 18],
}
union = sorted(set(occ["Enlace 0→1"]) | set(occ["Enlace 1→3"]))
rows = list(occ.items()) + [("Ruta (unión)", union)]
# First-Fit para 4 slots sobre la unión
free = [s not in union for s in range(n_slots)]
start = next(i for i in range(n_slots - 3) if all(free[i:i + 4]))
for r, (name, used) in enumerate(rows):
    y = len(rows) - 1 - r
    for s in range(n_slots):
        color = "#e9c46a" if s in used else "#f1f3f5"
        if r < 2 and start <= s < start + 4:
            color = "#2a9d8f"
        ax.add_patch(Rectangle((s, y), 0.92, 0.8, color=color, ec="#adb5bd", lw=0.5))
    ax.text(-0.4, y + 0.4, name, ha="right", va="center", fontsize=9)
ax.annotate("First-Fit: primer bloque de 4 slots\nlibre en TODOS los enlaces", xy=(start + 2, 2.85),
            xytext=(start + 7.5, 3.25), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#2a9d8f"))
for s in range(0, n_slots, 4):
    ax.text(s + 0.45, -0.35, str(s), ha="center", fontsize=7, color="#666")
ax.set_xlim(-5.2, n_slots + 0.2)
ax.set_ylim(-1.2, 3.9)
ax.axis("off")
for i, (lab, col) in enumerate([("ocupado", "#e9c46a"), ("libre", "#f1f3f5"), ("asignado a la nueva conexión", "#2a9d8f")]):
    ax.add_patch(Rectangle((i * 4.5, -1.05), 0.7, 0.4, color=col, ec="#adb5bd"))
    ax.text(i * 4.5 + 0.9, -0.85, lab, fontsize=8, va="center")
fig.tight_layout()
fig.savefig(OUT / "espectro.png", dpi=200)
plt.close(fig)

# 3. Gamma: peso de recompensas futuras ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.8))
t = np.arange(0, 60)
for g, c in [(0.99, "#e76f51"), (0.987, "#f4a261"), (0.83, "#2a9d8f")]:
    ax.plot(t, g ** t, color=c, lw=2, label=f"γ = {g}  (horizonte ≈ {1 / (1 - g):.0f} solicitudes)")
ax.set_xlabel("Solicitudes hacia el futuro (t)")
ax.set_ylabel("Peso γᵗ de la recompensa")
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "gamma.png", dpi=200)
plt.close(fig)

# 4. Diagramas de bloques ---------------------------------------------------------
def boxes(filename, items, arrows, size, title=None):
    fig, ax = plt.subplots(figsize=size)
    for key, (x, y, w, h, text, color) in items.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec="#343a40", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)
    for a, b, label in arrows:
        xa, ya, wa, ha_, _, _ = items[a]
        xb, yb, wb, hb, _, _ = items[b]
        start = (xa + wa / 2, ya + ha_ / 2)
        end = (xb + wb / 2, yb + hb / 2)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#343a40", lw=1.2,
                    shrinkA=28, shrinkB=28))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, fontsize=7.6,
                    ha="center", color="#495057", style="italic")
    ax.set_xlim(0, size[0]); ax.set_ylim(0, size[1]); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)

A, B, C, D = "#d8f3dc", "#dbe9f6", "#fde2c8", "#efe1f7"
boxes("bucle_rl.png", {
    "agent": (0.4, 1.3, 2.6, 1.2, "Agente PPO\n(red neuronal 64×64)\nelige ruta 0, 1 o 2", A),
    "env": (5.0, 1.3, 2.9, 1.2, "RMLSAEnv\n(nuestro entorno Gymnasium)\naplica ruta + modulación + FF", B),
    "sim": (5.0, 3.3, 2.9, 1.0, "Simulador DREAM-ON GYM\n(simNetPy: eventos, red, slots)", C),
}, [("agent", "env", "acción"), ("env", "sim", ""), ], (8.4, 4.6))
# flechas de vuelta dibujadas a mano

fig, ax = plt.subplots(figsize=(8.4, 4.4))
def box(x, y, w, h, text, color, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#343a40"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(p, q, label="", dy=0.13, color="#343a40"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, fontsize=7.8, ha="center", style="italic", color="#495057")
box(0.3, 1.6, 2.5, 1.3, "Agente PPO\nred neuronal 64×64\n(Stable-Baselines3)", A)
box(5.3, 1.6, 2.8, 1.3, "RMLSAEnv\nentorno Gymnasium\n(src/rmlsa/env.py)", B)
box(5.3, 3.4, 2.8, 0.85, "Simulador DREAM-ON GYM\nsimNetPy: eventos, red, slots", C)
box(5.3, 0.15, 2.8, 0.85, "Allocator first_fit\nruta elegida + modulación + FF", D)
arrow((2.8, 2.55), (5.3, 2.55), "acción: ruta 0, 1 o 2")
arrow((5.3, 1.95), (2.8, 1.95), "observación (47 valores) + recompensa ±1", dy=-0.28)
arrow((6.5, 2.9), (6.5, 3.4), "")
arrow((6.9, 3.4), (6.9, 2.9), "")
arrow((6.7, 1.6), (6.7, 1.0), "")
ax.text(7.05, 3.12, "step / estado", fontsize=7.5, style="italic", color="#495057")
ax.text(6.85, 1.25, "asigna slots", fontsize=7.5, style="italic", color="#495057")
ax.set_xlim(0, 8.4); ax.set_ylim(0, 4.4); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "bucle_rl.png", dpi=200); plt.close(fig)

# Pipeline general
fig, ax = plt.subplots(figsize=(8.6, 2.5))
steps = [("1. Escenario\nNSFNet, tráfico,\nbitrates", A), ("2. Baseline\nSPFF + PPO\npor defecto", B),
         ("3. Optuna\n100 trials\nTPE + poda", C), ("4. Evaluación\n3 entrenam. ×\n10 cargas × 10 sem.", D),
         ("5. Importancia\n+ análisis\nde la red", "#fff3bf")]
for i, (t_, c) in enumerate(steps):
    box(0.15 + i * 1.72, 0.6, 1.45, 1.3, t_, c, fs=8.3)
    if i:
        arrow((0.15 + i * 1.72 - 0.27, 1.25), (0.15 + i * 1.72, 1.25))
ax.text(0.15, 0.2, "Obj. 1", fontsize=8, color="#666"); ax.text(1.87, 0.2, "Obj. 1 y 4", fontsize=8, color="#666")
ax.text(3.59, 0.2, "Obj. 2 y 3", fontsize=8, color="#666"); ax.text(5.31, 0.2, "Obj. 4", fontsize=8, color="#666")
ax.text(7.03, 0.2, "Obj. 5", fontsize=8, color="#666")
ax.set_xlim(0, 8.6); ax.set_ylim(0, 2.2); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "pipeline.png", dpi=200); plt.close(fig)

# Evento discreto
fig, ax = plt.subplots(figsize=(8.6, 2.3))
ax.axhline(1, color="#343a40", lw=1)
evs = [(0.6, "A", "llega #1"), (1.5, "A", "llega #2"), (2.3, "D", "sale #1"), (3.2, "A", "llega #3"),
       (4.4, "A", "llega #4"), (5.0, "D", "sale #3"), (6.2, "A", "llega #5"), (7.4, "D", "sale #2")]
for x, kind, label in evs:
    c = "#2a9d8f" if kind == "A" else "#e76f51"
    ax.scatter(x, 1, s=70, color=c, zorder=3)
    ax.text(x, 1.25 if kind == "A" else 0.55, label, ha="center", fontsize=8, color=c)
ax.text(8.2, 0.95, "tiempo", fontsize=8.5)
ax.text(0.1, 1.75, "Llegadas (verde): el agente decide la ruta. Salidas (rojo): se liberan slots, sin decisión.",
        fontsize=8.5)
ax.set_xlim(0, 8.6); ax.set_ylim(0.2, 2.0); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "eventos.png", dpi=200); plt.close(fig)

# 5. Gráficos extra de Optuna v2 (slice y contorno) ------------------------------
sys.path.insert(0, str(REPO / "src"))
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name="ppo_rmlsa_100erl_v2", storage=f"sqlite:///{REPO / 'results/optuna.db'}")
optuna.visualization.matplotlib.plot_slice(study, params=["gamma", "learning_rate", "batch_size"])
plt.gcf().set_size_inches(11, 3.6)
plt.savefig(OUT / "slice.png", dpi=170, bbox_inches="tight"); plt.close("all")
done = [t for t in study.trials if t.state.name == "COMPLETE"]
pr = [t for t in study.trials if t.state.name == "PRUNED"]
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.scatter([t.params["gamma"] for t in pr], [t.params["learning_rate"] for t in pr], marker="x",
           color="#868e96", label="podado")
sc = ax.scatter([t.params["gamma"] for t in done], [t.params["learning_rate"] for t in done],
                c=[t.value for t in done], cmap="viridis_r", s=60, edgecolor="k", lw=0.5)
b = study.best_trial
ax.scatter(b.params["gamma"], b.params["learning_rate"], s=260, facecolor="none", edgecolor="red", lw=2,
           label=f"mejor (trial {b.number})")
ax.set_yscale("log"); ax.set_xlabel("gamma"); ax.set_ylabel("learning_rate (escala log)")
fig.colorbar(sc, label="probabilidad de bloqueo"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("Optuna v2: dónde probó TPE y qué tan bueno fue cada punto", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "contorno.png", dpi=200); plt.close(fig)
optuna.visualization.matplotlib.plot_intermediate_values(study)
plt.gcf().set_size_inches(7, 4)
ax = plt.gca(); ax.get_legend() and ax.get_legend().remove()
plt.savefig(OUT / "intermedios.png", dpi=170, bbox_inches="tight"); plt.close("all")
print("ok", sorted(p.name for p in OUT.iterdir()))
