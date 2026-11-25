"""Prueba rápida: SPFF, PPO y Optuna. Uso: python scripts/smoke_test.py"""

import contextlib
import io
import time

import optuna
from stable_baselines3 import PPO

from rmlsa.env import RMLSAEnv, evaluate_model, run_heuristic
from rmlsa.heuristics import spff

LOAD = 100
TRAIN_STEPS = 2048
EVAL_REQUESTS = 1000


def train(**hyperparams):
    env = RMLSAEnv(load=LOAD, seed=0)
    model = PPO("MlpPolicy", env, n_steps=512, verbose=0, **hyperparams)
    model.learn(total_timesteps=TRAIN_STEPS)
    return model


def objective(trial):
    model = train(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.9, 0.9999),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
    )
    return evaluate_model(model, LOAD, seed=1000, n_requests=EVAL_REQUESTS)


if __name__ == "__main__":
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        bp_spff = run_heuristic(spff, LOAD, seed=1000, n_requests=EVAL_REQUESTS)
        bp_ppo = evaluate_model(train(), LOAD, seed=1000, n_requests=EVAL_REQUESTS)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)

    print(f"[1/3] SPFF -> bloqueo = {bp_spff:.4f}")
    print(f"[2/3] PPO por defecto ({TRAIN_STEPS} pasos) -> bloqueo = {bp_ppo:.4f}")
    print(f"[3/3] Optuna (2 trials) -> mejor bloqueo = {study.best_value:.4f}, params = {study.best_params}")
    print(f"OK en {time.time() - t0:.0f} s")
