"""Verifica que la instalación funciona de punta a punta (~1-2 min en CPU).

1. Simula SPFF en NSFNet y reporta su probabilidad de bloqueo.
2. Entrena PPO (Stable-Baselines3) unos pocos pasos sobre DREAM-ON GYM.
3. Corre un estudio Optuna mínimo (2 trials) sobre lr, gamma y batch_size.

Uso:  python scripts/smoke_test.py
"""

import time

import optuna
from stable_baselines3 import PPO

from rmlsa.env import blocking_probability, make_env
from rmlsa.heuristics import spff

TRAIN_STEPS = 2048
EVAL_STEPS = 1000


def evaluate(model, seed):
    env = make_env(seed=seed)
    env.start()
    obs, _ = env.reset()
    for _ in range(EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, *_ = env.step(action)
    return blocking_probability(env)


def run_spff(seed):
    env = make_env(seed=seed, allocator=spff)
    env.start()
    for _ in range(EVAL_STEPS):
        env.step(0)
    return blocking_probability(env)


def objective(trial):
    env = make_env(seed=1)
    env.start()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.9, 0.9999),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        n_steps=512,
        verbose=0,
    )
    model.learn(total_timesteps=TRAIN_STEPS)
    return evaluate(model, seed=100)


if __name__ == "__main__":
    t0 = time.time()
    print(f"[1/3] SPFF  -> bloqueo = {run_spff(seed=100):.4f}")

    env = make_env(seed=1)
    env.start()
    model = PPO("MlpPolicy", env, n_steps=512, verbose=0)
    model.learn(total_timesteps=TRAIN_STEPS)
    print(f"[2/3] PPO por defecto ({TRAIN_STEPS} pasos) -> bloqueo = {evaluate(model, 100):.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    print(f"[3/3] Optuna (2 trials) -> mejor bloqueo = {study.best_value:.4f}, params = {study.best_params}")

    print(f"OK en {time.time() - t0:.0f} s")
