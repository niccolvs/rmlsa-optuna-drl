"""Optimiza learning_rate, gamma y batch_size de PPO con Optuna (TPE + MedianPruner).

Cada trial entrena un PPO por etapas y, al final de cada etapa, mide su probabilidad de
bloqueo. Si va peor que la mediana de los trials anteriores en esa misma etapa, se poda.
El estudio se guarda en results/optuna.db: si se corta (Ctrl + C), al relanzar el mismo
comando continúa donde quedó.

Uso:
    python scripts/optuna_ppo.py
    python scripts/optuna_ppo.py --n-trials 40 --load 100

Al terminar guarda en results/optuna/<study-name>/: historial, importancia de hiperparámetros y trials.csv.
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
EVAL_SEED = 1000  # tráfico distinto al de entrenamiento (semilla 0)


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
            with contextlib.redirect_stdout(io.StringIO()):
                blocking = evaluate_model(model, args.load, EVAL_SEED, args.eval_requests)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=40, help="total de trials del estudio")
    parser.add_argument("--load", type=float, default=100, help="carga en Erlang")
    parser.add_argument("--timesteps", type=int, default=100_000, help="pasos de entrenamiento por trial")
    parser.add_argument("--stages", type=int, default=5, help="evaluaciones intermedias (para la poda)")
    parser.add_argument("--eval-requests", type=int, default=10_000)
    parser.add_argument("--study-name", default="ppo_rmlsa_100erl_v2")
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
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - done)
    print(f"Estudio '{args.study_name}': {done} trials previos, se harán {remaining} más (total {args.n_trials}).")
    print(f"Cada trial: {args.timesteps} pasos a {args.load} Erlang (~2.5 min si no se poda).\n")

    t0 = time.time()
    study.optimize(make_objective(args), n_trials=remaining, callbacks=[print_trial])
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min")

    states = [t.state for t in study.trials]
    print(f"Completos: {states.count(optuna.trial.TrialState.COMPLETE)}  "
          f"Podados: {states.count(optuna.trial.TrialState.PRUNED)}")
    print(f"Mejor bloqueo: {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    out = RESULTS / "optuna" / args.study_name
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(out / "historial.png", dpi=150, bbox_inches="tight")
    importances = optuna.importance.get_param_importances(study)
    print("Importancia de hiperparámetros:", {k: round(v, 3) for k, v in importances.items()})
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(out / "importancia.png", dpi=150, bbox_inches="tight")
    print(f"Gráficos y trials.csv en {out}")


if __name__ == "__main__":
    main()
