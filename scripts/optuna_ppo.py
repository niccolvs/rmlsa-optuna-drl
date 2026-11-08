"""Optimiza learning_rate, gamma y batch_size de PPO con Optuna (TPE + MedianPruner).

Cada trial entrena un PPO por etapas y, al final de cada etapa, mide su probabilidad de
bloqueo (promedio sobre las semillas de tráfico de --eval-seeds). Si va peor que la mediana
de los trials anteriores en esa misma etapa, se poda.
El estudio se guarda en results/optuna.db: si se corta (Ctrl + C), al relanzar el mismo
comando continúa donde quedó.

Uso:
    python scripts/optuna_ppo.py
    python scripts/optuna_ppo.py --n-trials 40 --timesteps 100000 --eval-seeds 1000 --study-name ppo_rmlsa_100erl_v2

Al terminar guarda en results/optuna/<study-name>/: trials.csv y los gráficos de Optuna (historial,
importancia, slice, coordenadas paralelas, valores intermedios) más gamma vs learning rate.
"""

import argparse
import contextlib
import io
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from stable_baselines3 import PPO

from rmlsa.env import RMLSAEnv, evaluate_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_objective(args):
    def objective(trial):
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.8, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        }
        with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido
            env = RMLSAEnv(load=args.load, seed=0)
            env.reset()  # construye el simulador (warm-up) aquí y no dentro de learn()
            model = PPO("MlpPolicy", env, seed=0, verbose=0, **hyperparams)

        steps_per_stage = args.timesteps // args.stages
        for stage in range(args.stages):
            model.learn(steps_per_stage, reset_num_timesteps=False)
            with contextlib.redirect_stdout(io.StringIO()):  # tráfico distinto al de entrenamiento (semilla 0)
                blocking = sum(evaluate_model(model, args.load, s, args.eval_requests)
                               for s in args.eval_seeds) / len(args.eval_seeds)
            trial.report(blocking, stage)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return blocking

    return objective


def print_trial(study, trial):
    n = len(study.trials)
    if trial.state == optuna.trial.TrialState.PRUNED:
        estado = "podado"
    else:
        estado = f"bloqueo = {trial.value:.4f}"
    params = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.params.items())
    mejor = f"{study.best_value:.4f}" if any(t.value is not None for t in study.trials) else "-"
    print(f"[trial {n:>3}] {estado:18s} | {params}  | mejor hasta ahora: {mejor}")


def plot_gamma_lr(study, path):
    """Dónde probó TPE en el plano gamma-learning rate, coloreado por bloqueo (podados con X)."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter([t.params["gamma"] for t in pruned], [t.params["learning_rate"] for t in pruned],
               marker="x", color="#868e96", label="podado")
    sc = ax.scatter([t.params["gamma"] for t in done], [t.params["learning_rate"] for t in done],
                    c=[t.value for t in done], cmap="viridis_r", s=55, edgecolor="k", linewidth=0.5)
    best = study.best_trial
    ax.scatter(best.params["gamma"], best.params["learning_rate"], s=250, facecolor="none",
               edgecolor="red", linewidth=2, label=f"mejor (trial {best.number})")
    ax.set_yscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("learning_rate")
    fig.colorbar(sc, label="probabilidad de bloqueo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def save_plots(study, out):
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    vis = optuna.visualization.matplotlib
    vis.plot_optimization_history(study)
    plt.savefig(out / "historial.png", dpi=150, bbox_inches="tight")
    vis.plot_param_importances(study)
    plt.savefig(out / "importancia.png", dpi=150, bbox_inches="tight")
    vis.plot_slice(study)
    plt.gcf().set_size_inches(11, 3.6)
    plt.savefig(out / "slice.png", dpi=150, bbox_inches="tight")
    vis.plot_parallel_coordinate(study)
    plt.savefig(out / "coordenadas_paralelas.png", dpi=150, bbox_inches="tight")
    vis.plot_intermediate_values(study)
    legend = plt.gca().get_legend()
    if legend:
        legend.remove()  # una entrada por trial: ilegible
    plt.savefig(out / "valores_intermedios.png", dpi=150, bbox_inches="tight")
    plot_gamma_lr(study, out / "gamma_vs_learning_rate.png")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100, help="total de trials del estudio")
    parser.add_argument("--load", type=float, default=100, help="carga en Erlang")
    parser.add_argument("--timesteps", type=int, default=200_000, help="pasos de entrenamiento por trial")
    parser.add_argument("--stages", type=int, default=5, help="evaluaciones intermedias (para la poda)")
    parser.add_argument("--eval-requests", type=int, default=10_000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 1001, 1002],
                        help="semillas de tráfico con que se mide cada trial (se promedia)")
    parser.add_argument("--study-name", default="ppo_rmlsa_100erl_v3")
    parser.add_argument("--solo-graficos", action="store_true",
                        help="no entrena: solo regenera los gráficos de un estudio ya hecho")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{RESULTS / 'optuna.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    out = RESULTS / "optuna" / args.study_name
    if args.solo_graficos:
        save_plots(study, out)
        print(f"Gráficos en {out}")
        return

    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - done)
    print(f"Estudio '{args.study_name}': {done} trials previos, se harán {remaining} más (total {args.n_trials}).")
    print(f"Cada trial: {args.timesteps} pasos a {args.load} Erlang, medido con semillas {args.eval_seeds}.\n")

    t0 = time.time()
    study.optimize(make_objective(args), n_trials=remaining, callbacks=[print_trial])
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min")

    states = [t.state for t in study.trials]
    print(f"Completos: {states.count(optuna.trial.TrialState.COMPLETE)}  "
          f"Podados: {states.count(optuna.trial.TrialState.PRUNED)}")
    print(f"Mejor bloqueo: {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")
    importances = optuna.importance.get_param_importances(study)
    print("Importancia de hiperparámetros:", {k: round(v, 3) for k, v in importances.items()})
    save_plots(study, out)
    print(f"Gráficos y trials.csv en {out}")


if __name__ == "__main__":
    main()
