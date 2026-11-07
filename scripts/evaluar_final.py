"""Comparación final (objetivo 4): PPO por defecto vs PPO optimizado vs SPFF.

1. Entrena PPO con hiperparámetros por defecto y con los mejores del estudio Optuna,
   cada uno con varias semillas de entrenamiento (el RL varía de una corrida a otra).
2. Evalúa cada agente y SPFF con varias cargas y semillas de tráfico de prueba, distintas
   de la usada por Optuna (1000), para no medir sobre el mismo tráfico con que se eligió.

Los modelos ya entrenados se reutilizan: si se corta, al relanzar sigue donde quedó.

Uso:
    python scripts/evaluar_final.py
    python scripts/evaluar_final.py --study-name ppo_rmlsa_100erl_v2 --timesteps 300000

Guarda en results/final/: resultados.csv, resumen.csv y bloqueo_vs_carga.png.
"""

import argparse
import contextlib
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from stable_baselines3 import PPO

from rmlsa.env import RMLSAEnv, evaluate_model, run_heuristic
from rmlsa.heuristics import spff

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIN_LOAD = 100  # carga con la que se entrenan los agentes y se corrió Optuna


def train(name, hyperparams, seed, timesteps):
    path = RESULTS / "models" / "final" / f"{name}_s{seed}.zip"
    if path.exists():
        print(f"  {path.stem}: ya entrenado, se reutiliza")
        return PPO.load(path)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
        env = RMLSAEnv(load=TRAIN_LOAD, seed=seed)
        env.reset()
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                tensorboard_log=str(RESULTS / "tensorboard"), **hyperparams)
    model.learn(total_timesteps=timesteps, tb_log_name=f"final_{name}_s{seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"  {path.stem}: entrenado en {(time.time() - t0) / 60:.1f} min")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", default="ppo_rmlsa_100erl_v2")
    parser.add_argument("--timesteps", type=int, default=300_000, help="pasos de entrenamiento por agente")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--loads", type=float, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[2000, 2001, 2002, 2003, 2004])
    parser.add_argument("--requests", type=int, default=10_000, help="solicitudes por evaluación")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{RESULTS / 'optuna.db'}")
    configs = {"PPO defecto": {}, "PPO Optuna": study.best_params}
    print(f"Hiperparámetros de Optuna ({args.study_name}): {study.best_params}\n")

    t0 = time.time()
    print(f"[1/2] Entrenando {len(configs) * len(args.train_seeds)} agentes "
          f"({args.timesteps} pasos cada uno, {TRAIN_LOAD} Erlang)")
    models = {}
    for name, hyperparams in configs.items():
        slug = "optuna" if hyperparams else "defecto"
        for seed in args.train_seeds:
            models[(name, seed)] = train(slug, hyperparams, seed, args.timesteps)

    print(f"\n[2/2] Evaluando con cargas {args.loads} y semillas de tráfico {args.test_seeds}")
    rows = []
    for load in args.loads:
        for test_seed in args.test_seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                rows.append({"agente": "SPFF", "semilla_entrenamiento": None, "carga_erlang": load,
                             "semilla_trafico": test_seed,
                             "bloqueo": run_heuristic(spff, load, test_seed, args.requests)})
                for (name, seed), model in models.items():
                    rows.append({"agente": name, "semilla_entrenamiento": seed, "carga_erlang": load,
                                 "semilla_trafico": test_seed,
                                 "bloqueo": evaluate_model(model, load, test_seed, args.requests)})
        print(f"  carga {load:>5.0f} Erl lista")

    out = RESULTS / "final"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "resultados.csv", index=False)

    summary = df.groupby(["agente", "carga_erlang"])["bloqueo"].agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "resumen.csv", index=False)
    table = summary.pivot(index="carga_erlang", columns="agente", values="mean")[["SPFF", "PPO defecto", "PPO Optuna"]]
    print("\nProbabilidad de bloqueo promedio:")
    print(table.round(4).to_string())
    print("\nMejora de PPO Optuna respecto de SPFF:")
    print(((1 - table["PPO Optuna"] / table["SPFF"]) * 100).round(1).astype(str).add(" %").to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, marker in [("SPFF", "s"), ("PPO defecto", "o"), ("PPO Optuna", "^")]:
        g = summary[summary["agente"] == name]
        ax.errorbar(g["carga_erlang"], g["mean"], yerr=g["std"], marker=marker, capsize=3, label=name)
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_yscale("log")
    ax.set_title(f"NSFNet: {len(args.train_seeds)} entrenamientos × {len(args.test_seeds)} semillas de tráfico")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bloqueo_vs_carga.png", dpi=150)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min. Resultados en {out}")


if __name__ == "__main__":
    main()
