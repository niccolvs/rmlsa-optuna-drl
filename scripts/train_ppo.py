"""Entrena un agente PPO con los hiperparámetros POR DEFECTO de Stable-Baselines3.

Uso:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --load 100 --timesteps 200000 --seed 0 --name ppo_default

Durante el entrenamiento se registran en TensorBoard (results/tensorboard):
    rmlsa/bloqueo_episodio   probabilidad de bloqueo de cada episodio (1000 solicitudes)
    rollout/ep_rew_mean      recompensa media (+1 asignada, -1 bloqueada)
    train/*                  pérdidas y métricas internas de PPO
Al final guarda el modelo en results/models/<name>.zip y lo compara con SPFF.
"""

import argparse
import contextlib
import io
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rmlsa.env import RMLSAEnv, evaluate_model, run_heuristic
from rmlsa.heuristics import spff

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_SEED = 1000  # semilla de tráfico distinta a la de entrenamiento


class BlockingLogger(BaseCallback):
    """Registra la probabilidad de bloqueo de cada episodio en TensorBoard."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "blocking_probability" in info:
                self.logger.record_mean("rmlsa/bloqueo_episodio", info["blocking_probability"])
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", type=float, default=100, help="carga en Erlang")
    parser.add_argument("--timesteps", type=int, default=200_000, help="solicitudes de entrenamiento")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="ppo_default")
    parser.add_argument("--eval-requests", type=int, default=10_000)
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):  # el simulador imprime ruido al iniciar
        env = RMLSAEnv(load=args.load, seed=args.seed)
        env.reset()

    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=str(RESULTS / "tensorboard"))
    print(f"Entrenando {args.name}: {args.timesteps} pasos, carga {args.load} Erlang ...")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=BlockingLogger(), tb_log_name=args.name)
    print(f"Entrenamiento listo en {(time.time() - t0) / 60:.1f} min")

    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    model.save(RESULTS / "models" / args.name)

    print(f"\nEvaluación ({args.eval_requests} solicitudes, semilla de tráfico {EVAL_SEED}):")
    with contextlib.redirect_stdout(io.StringIO()):
        results = {
            "PPO": evaluate_model(model, args.load, EVAL_SEED, args.eval_requests),
            "SPFF": run_heuristic(spff, args.load, EVAL_SEED, args.eval_requests),
        }
    for name, bp in results.items():
        print(f"  {name:7s} bloqueo = {bp:.4f}")
    print(f"\nModelo guardado en {RESULTS / 'models' / (args.name + '.zip')}")


if __name__ == "__main__":
    main()
