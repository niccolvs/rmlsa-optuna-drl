"""Probabilidad de bloqueo de SPFF y KSP-FF bajo distintas cargas y semillas.

Uso:
    python scripts/baseline_heuristicas.py
    python scripts/baseline_heuristicas.py --loads 50 100 150 200 --seeds 0 1 2 3 4

Guarda results/baseline_heuristicas.csv y results/baseline_heuristicas.png.
"""

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rmlsa.env import run_heuristic
from rmlsa.heuristics import ksp_ff, spff

RESULTS = Path(__file__).resolve().parents[1] / "results"
HEURISTICS = {"SPFF": spff, "KSP-FF": ksp_ff}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loads", type=float, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--requests", type=int, default=10000, help="solicitudes evaluadas por corrida")
    args = parser.parse_args()

    rows = []
    for load in args.loads:
        for seed in args.seeds:
            for name, allocator in HEURISTICS.items():
                with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
                    bp = run_heuristic(allocator, load, seed, args.requests)
                rows.append({"heuristica": name, "carga_erlang": load, "semilla": seed, "bloqueo": bp})
                print(f"{name:7s} carga={load:>5.0f} Erl  semilla={seed}  bloqueo={bp:.4f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "baseline_heuristicas.csv", index=False)

    summary = df.groupby(["heuristica", "carga_erlang"])["bloqueo"].agg(["mean", "std"]).reset_index()
    print("\nPromedio ± desviación estándar entre semillas:")
    print(summary.pivot(index="carga_erlang", columns="heuristica", values="mean").round(4))

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in summary.groupby("heuristica"):
        ax.errorbar(g["carga_erlang"], g["mean"], yerr=g["std"], marker="o", capsize=3, label=name)
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_yscale("log")
    ax.set_title(f"NSFNet, {args.requests} solicitudes, {len(args.seeds)} semillas")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_heuristicas.png", dpi=150)
    print(f"\nGuardado en {RESULTS / 'baseline_heuristicas.csv'} y .png")


if __name__ == "__main__":
    main()
