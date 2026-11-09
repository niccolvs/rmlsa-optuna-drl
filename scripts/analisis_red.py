"""Gráficos del análisis de la red a partir de lo que guardó evaluar_final.py.

    curvas_aprendizaje.png   bloqueo por episodio durante el entrenamiento (media y rango de las semillas)
    uso_rutas.png            % de solicitudes enviadas por una ruta que no es la más corta, según la carga
    bloqueo_por_bitrate.png  bloqueo de cada velocidad (10/40/100/400 Gbps) a la carga indicada
    bloqueo_por_distancia.png  bloqueo según el largo de la ruta más corta del par origen-destino
    mapa_ocupacion.png       NSFNet coloreada por ocupación media de cada enlace: SPFF vs PPO Optuna

Uso:
    python scripts/analisis_red.py
    python scripts/analisis_red.py --run ppo_rmlsa_100erl_v3 --carga 100
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ["SPFF", "PPO defecto", "PPO Optuna"]
COLORS = {"SPFF": "#1f77b4", "PPO defecto": "#ff7f0e", "PPO Optuna": "#2ca02c"}
# Posiciones aproximadas de los nodos de NSFNet (solo para dibujar)
POS = {0: (1, 9), 1: (0.4, 6.2), 2: (1.3, 3.4), 3: (3.2, 6.6), 4: (4.8, 5.6), 5: (6.2, 2.4), 6: (6.2, 6.6),
       7: (7.7, 7.3), 8: (10.0, 6.4), 9: (9.2, 2.8), 10: (8.9, 8.7), 11: (11.4, 8.3), 12: (10.9, 7.4), 13: (12.2, 5.0)}


def learning_curves(out):
    files = sorted((out / "curvas").glob("*.csv"))
    if not files:
        print("  (sin curvas de aprendizaje)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for slug, name in [("defecto", "PPO defecto"), ("optuna", "PPO Optuna")]:
        curves = []
        for f in files:
            if f.stem.startswith(slug + "_"):
                c = pd.read_csv(f)
                curves.append(c["bloqueo"].rolling(10, min_periods=1).mean().to_numpy())
        if not curves:
            continue
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        steps = pd.read_csv(next(f for f in files if f.stem.startswith(slug + "_")))["paso"].to_numpy()[:n]
        ax.plot(steps, arr.mean(0), color=COLORS[name], label=f"{name} (media de {len(curves)})")
        ax.fill_between(steps, arr.min(0), arr.max(0), color=COLORS[name], alpha=0.2)
    ax.set_xlabel("Pasos de entrenamiento (solicitudes)")
    ax.set_ylabel("Bloqueo por episodio (media móvil de 10)")
    ax.set_title("Curvas de aprendizaje a 100 Erlang\n(banda = mínimo y máximo entre semillas)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curvas_aprendizaje.png", dpi=150)
    plt.close(fig)


def route_usage(out):
    r = pd.read_csv(out / "red_rutas.csv")
    r = r[r["agente"] != "SPFF"]
    tot = r.groupby(["agente", "carga_erlang"])["solicitudes"].sum()
    desv = r[r["clave"] != 0].groupby(["agente", "carga_erlang"])["solicitudes"].sum().reindex(tot.index, fill_value=0)
    pct = (desv / tot * 100).reset_index(name="pct")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ["PPO defecto", "PPO Optuna"]:
        g = pct[pct["agente"] == name]
        ax.plot(g["carga_erlang"], g["pct"], marker="o", color=COLORS[name], label=name)
    ax.axhline(0, color=COLORS["SPFF"], lw=1.5, ls="--", label="SPFF (nunca se desvía)")
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Solicitudes por ruta alternativa [%]")
    ax.set_title("¿Cuánto se desvía el agente de la ruta más corta?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "uso_rutas.png", dpi=150)
    plt.close(fig)
    return pct


def blocking_by(out, kind, load, bins=None, labels=None):
    d = pd.read_csv(out / f"red_{kind}.csv")
    d = d[d["carga_erlang"] == load]
    if bins is not None:
        d["clave"] = pd.cut(d["clave"], bins=bins, labels=labels)
    g = d.groupby(["agente", "clave"], observed=True)[["solicitudes", "bloqueadas"]].sum()
    g["bloqueo"] = g["bloqueadas"] / g["solicitudes"]
    table = g["bloqueo"].unstack(0)[AGENTS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(table))
    for i, name in enumerate(AGENTS):
        ax.bar(x + (i - 1) * 0.27, table[name], width=0.27, color=COLORS[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} Gbps" for k in table.index] if kind == "bitrate" else list(table.index))
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_xlabel("Velocidad pedida" if kind == "bitrate" else "Largo de la ruta más corta del par [km]")
    ax.set_title(f"Bloqueo por {'velocidad' if kind == 'bitrate' else 'distancia'} a {load:.0f} Erlang")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"bloqueo_por_{kind}.png", dpi=150)
    plt.close(fig)
    return table


def occupancy_map(out, load):
    topo = json.load(open(ROOT / "data" / "topologies" / "NSFNet.json"))
    ends = {l["id"]: (l["src"], l["dst"]) for l in topo["links"]}
    d = pd.read_csv(out / "red_enlaces.csv")
    d = d[d["carga_erlang"] == load]
    occ = d.groupby(["agente", "clave"])["solicitudes"].mean()  # columna 'solicitudes' = ocupación media
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    vmax = occ.max()
    for ax, name in zip(axes, ["SPFF", "PPO Optuna"]):
        pair = {}
        for lid, (a, b) in ends.items():
            pair.setdefault((min(a, b), max(a, b)), []).append(occ[(name, lid)])
        segs = [[POS[a], POS[b]] for a, b in pair]
        vals = [np.mean(v) for v in pair.values()]
        lc = LineCollection(segs, array=np.array(vals), cmap="RdYlGn_r", linewidths=6)
        lc.set_clim(0, vmax)
        ax.add_collection(lc)
        for n, (x, y) in POS.items():
            ax.scatter(x, y, s=260, color="#1d3557", zorder=3)
            ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=8, zorder=4)
        ax.set_title(f"{name}\nocupación media {np.mean(vals) * 100:.1f} %, máxima {max(vals) * 100:.0f} %")
        ax.set_xlim(-0.3, 12.9)
        ax.set_ylim(1.7, 9.7)
        ax.axis("off")
    fig.colorbar(lc, ax=axes, label="Fracción de slots ocupados", shrink=0.8)
    fig.suptitle(f"Ocupación del espectro por enlace a {load:.0f} Erlang (disposición aproximada)", y=1.06)
    fig.savefig(out / "mapa_ocupacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return occ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="ppo_rmlsa_100erl_v3", help="carpeta dentro de results/final")
    parser.add_argument("--carga", type=float, default=100)
    args = parser.parse_args()
    out = ROOT / "results" / "final" / args.run

    learning_curves(out)
    pct = route_usage(out)
    print("Solicitudes por ruta alternativa [%]:")
    print(pct.pivot(index="carga_erlang", columns="agente", values="pct").round(1).to_string())
    print(f"\nBloqueo por velocidad a {args.carga:.0f} Erlang:")
    print(blocking_by(out, "bitrate", args.carga).round(4).to_string())
    print(f"\nBloqueo por distancia a {args.carga:.0f} Erlang:")
    print(blocking_by(out, "distancia", args.carga, bins=[0, 1000, 2000, 3000, 10_000],
                      labels=["< 1000", "1000-2000", "2000-3000", "> 3000"]).round(4).to_string())
    occupancy_map(out, args.carga)
    print(f"\nGráficos en {out}")


if __name__ == "__main__":
    main()
