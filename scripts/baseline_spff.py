"""Probabilidad de bloqueo de SPFF bajo distintas cargas y semillas.

Uso:
    python scripts/baseline_spff.py
    python scripts/baseline_spff.py --loads 50 100 150 200 --seeds 0 1 2 3 4

Guarda results/baseline_spff.csv y results/baseline_spff.png.
"""

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rmlsa.env import run_heuristic
from rmlsa.heuristics import spff

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loads", type=float, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--requests", type=int, default=10000, help="solicitudes evaluadas por corrida")
    args = parser.parse_args()

    rows = []
    for load in args.loads:
        for seed in args.seeds:
            with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
                bp = run_heuristic(spff, load, seed, args.requests)
            rows.append({"carga_erlang": load, "semilla": seed, "bloqueo": bp})
            print(f"SPFF  carga={load:>5.0f} Erl  semilla={seed}  bloqueo={bp:.4f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "baseline_spff.csv", index=False)

    summary = df.groupby("carga_erlang")["bloqueo"].agg(["mean", "std"]).reset_index()
    print("\nPromedio ± desviación estándar entre semillas:")
    print(summary.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(summary["carga_erlang"], summary["mean"], yerr=summary["std"], marker="o", capsize=3, label="SPFF")
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_yscale("log")
    ax.set_title(f"NSFNet, {args.requests} solicitudes, {len(args.seeds)} semillas")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_spff.png", dpi=150)
    print(f"\nGuardado en {RESULTS / 'baseline_spff.csv'} y .png")


if __name__ == "__main__":
    main()
