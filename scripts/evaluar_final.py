"""Comparación final (objetivo 4): PPO por defecto vs PPO optimizado vs SPFF.

1. Entrena PPO con hiperparámetros por defecto y con los mejores del estudio Optuna,
   cada uno con varias semillas de entrenamiento (el RL varía de una corrida a otra).
   Guarda la curva de aprendizaje (bloqueo de cada episodio) de cada entrenamiento.
2. Evalúa cada agente y SPFF con varias cargas y semillas de tráfico de prueba, distintas
   de las usadas por Optuna (1000-1002), para no medir sobre el mismo tráfico con que se eligió.
   Además registra cómo usa la red cada agente: rutas elegidas, bloqueo por bitrate y por
   distancia, y ocupación de cada enlace.

Los modelos ya entrenados se reutilizan: si se corta, al relanzar sigue donde quedó.

Uso:
    python scripts/evaluar_final.py
    python scripts/evaluar_final.py --study-name ppo_rmlsa_100erl_v3 --timesteps 1000000 --tag largo

Guarda en results/final/<study-name>[_<tag>]/: resultados.csv, resumen.csv, red_*.csv, curvas/*.csv
y los gráficos bloqueo_vs_carga.png y mejora_vs_carga.png. Los gráficos del análisis de la red
los genera scripts/analisis_red.py.
"""

import argparse
import contextlib
import csv
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rmlsa.env import RMLSAEnv, evaluate_detailed

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIN_LOAD = 100  # carga con la que se entrenan los agentes y se corrió Optuna
AGENTS = ["SPFF", "PPO defecto", "PPO Optuna"]
# t de Student al 97.5 % según grados de libertad, para intervalos de confianza del 95 %
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        14: 2.14, 19: 2.09, 29: 2.05}


def t975(dof):
    return T975.get(dof) or T975[min((d for d in T975 if d >= dof), default=29)]


class CurveLogger(BaseCallback):
    """Guarda (paso, bloqueo del episodio) de cada episodio de entrenamiento."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "blocking_probability" in info:
                self.rows.append((self.num_timesteps, info["blocking_probability"]))
        return True


def train(slug, hyperparams, seed, timesteps, models_dir, curves_dir):
    path = models_dir / f"{slug}_s{seed}.zip"
    if path.exists():
        print(f"  {path.stem}: ya entrenado, se reutiliza")
        return PPO.load(path)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
        env = RMLSAEnv(load=TRAIN_LOAD, seed=seed)
        env.reset()
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                tensorboard_log=str(RESULTS / "tensorboard"), **hyperparams)
    logger = CurveLogger()
    model.learn(total_timesteps=timesteps, callback=logger, tb_log_name=f"{models_dir.name}_{slug}_s{seed}")
    models_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logger.rows, columns=["paso", "bloqueo"]).to_csv(curves_dir / f"{slug}_s{seed}.csv", index=False)
    model.save(path)  # al final: si se corta a medias, se vuelve a entrenar completo
    print(f"  {path.stem}: entrenado en {(time.time() - t0) / 60:.1f} min")
    return model


def summarize(df):
    """Por agente y carga: promedio sobre entrenamientos, luego media e IC 95 % sobre semillas de tráfico."""
    per_seed = df.groupby(["agente", "carga_erlang", "semilla_trafico"])["bloqueo"].mean().reset_index()
    g = per_seed.groupby(["agente", "carga_erlang"])["bloqueo"]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary["ic95"] = [t975(n - 1) * s / n ** 0.5 for s, n in zip(summary["std"], summary["count"])]
    return per_seed, summary


def improvement(per_seed, a, b):
    """Mejora porcentual de a respecto de b por carga, con IC 95 % (pareado por semilla de tráfico)."""
    p = per_seed.pivot_table(index=["carga_erlang", "semilla_trafico"], columns="agente", values="bloqueo")
    imp = ((1 - p[a] / p[b]) * 100).groupby(level=0)
    out = imp.agg(["mean", "std", "count"])
    out["ic95"] = [t975(n - 1) * s / n ** 0.5 for s, n in zip(out["std"], out["count"])]
    out["casos_mejor"] = (p[a] < p[b]).groupby(level=0).sum()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", default="ppo_rmlsa_100erl_v3")
    parser.add_argument("--tag", default="", help="sufijo para separar variantes (p. ej. 'largo')")
    parser.add_argument("--timesteps", type=int, default=300_000, help="pasos de entrenamiento por agente")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--loads", type=float, nargs="+", default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(2000, 2010)))
    parser.add_argument("--requests", type=int, default=10_000, help="solicitudes por evaluación")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{RESULTS / 'optuna.db'}")
    configs = {"PPO defecto": ("defecto", {}), "PPO Optuna": ("optuna", study.best_params)}
    print(f"Hiperparámetros de Optuna ({args.study_name}): {study.best_params}\n")
    run_name = args.study_name + (f"_{args.tag}" if args.tag else "")
    out = RESULTS / "final" / run_name
    models_dir = RESULTS / "models" / "final" / run_name

    t0 = time.time()
    print(f"[1/2] Entrenando {len(configs) * len(args.train_seeds)} agentes "
          f"({args.timesteps} pasos cada uno, {TRAIN_LOAD} Erlang)")
    models = {}
    for name, (slug, hyperparams) in configs.items():
        for seed in args.train_seeds:
            models[(name, seed)] = train(slug, hyperparams, seed, args.timesteps, models_dir, out / "curvas")

    print(f"\n[2/2] Evaluando con cargas {[int(x) for x in args.loads]} y semillas de tráfico "
          f"{args.test_seeds[0]}-{args.test_seeds[-1]} ({len(args.test_seeds)})")
    rows, red = [], {"rutas": [], "bitrate": [], "distancia": [], "enlaces": []}
    agents = [("SPFF", None, None)] + [(name, seed, m) for (name, seed), m in models.items()]
    for load in args.loads:
        t_load = time.time()
        for test_seed in args.test_seeds:
            for name, seed, model in agents:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = evaluate_detailed(load, test_seed, args.requests, model)
                key = {"agente": name, "semilla_entrenamiento": seed, "carga_erlang": load,
                       "semilla_trafico": test_seed}
                rows.append({**key, "bloqueo": r["bloqueo"]})
                for k, n in r["rutas"].items():
                    red["rutas"].append({**key, "clave": k, "solicitudes": n, "bloqueadas": ""})
                for kind in ("bitrate", "distancia"):
                    for k, (n, b) in r[kind].items():
                        red[kind].append({**key, "clave": k, "solicitudes": n, "bloqueadas": b})
                for k, occ in r["enlaces"].items():
                    red["enlaces"].append({**key, "clave": k, "solicitudes": round(occ, 5), "bloqueadas": ""})
        print(f"  carga {load:>5.0f} Erl lista ({(time.time() - t_load) / 60:.1f} min)")

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "resultados.csv", index=False)
    for kind, data in red.items():
        with open(out / f"red_{kind}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    per_seed, summary = summarize(df)
    summary.to_csv(out / "resumen.csv", index=False)
    table = summary.pivot(index="carga_erlang", columns="agente", values="mean")[AGENTS]
    print("\nProbabilidad de bloqueo promedio:")
    print(table.round(4).to_string())
    for a, b in [("PPO Optuna", "SPFF"), ("PPO Optuna", "PPO defecto"), ("PPO defecto", "SPFF")]:
        imp = improvement(per_seed, a, b)
        imp.to_csv(out / f"mejora_{a.replace(' ', '_')}_vs_{b.replace(' ', '_')}.csv")
        print(f"\nMejora de {a} respecto de {b} (media ± IC 95 %, semillas donde {a} bloquea menos):")
        for load, r in imp.iterrows():
            print(f"  {load:>5.0f} Erl: {r['mean']:5.1f} % ± {r['ic95']:.1f}   ({int(r['casos_mejor'])}/{int(r['count'])})")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, marker in zip(AGENTS, ["s", "o", "^"]):
        g = summary[summary["agente"] == name]
        ax.errorbar(g["carga_erlang"], g["mean"], yerr=g["ic95"], marker=marker, capsize=3, label=name)
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_yscale("log")
    ax.set_title(f"NSFNet: {len(args.train_seeds)} entrenamientos × {len(args.test_seeds)} semillas (IC 95 %)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bloqueo_vs_carga.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (a, b), marker in zip([("PPO Optuna", "SPFF"), ("PPO Optuna", "PPO defecto"), ("PPO defecto", "SPFF")],
                              ["^", "D", "o"]):
        imp = improvement(per_seed, a, b)
        ax.errorbar(imp.index, imp["mean"], yerr=imp["ic95"], marker=marker, capsize=3, label=f"{a} vs {b}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Carga de tráfico [Erlang]")
    ax.set_ylabel("Reducción del bloqueo [%]")
    ax.set_title("Mejora relativa (IC 95 %)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mejora_vs_carga.png", dpi=150)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min. Resultados en {out}")


if __name__ == "__main__":
    main()
