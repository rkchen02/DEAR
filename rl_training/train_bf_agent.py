import argparse
import time
from pathlib import Path
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.bellman_ford_env import BellmanFordEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Bellman-FordEnv.")
    parser.add_argument("--n-nodes", type=int, default=16)
    parser.add_argument("--reward-mode", type=str, choices=["dense", "sparse"], default="sparse")

    parser.add_argument(
        "--no-clrs",
        action="store_true",
        help="Disable CLRS graphs and use random graphs instead.",
    )
    parser.add_argument(
        "--clrs-root",
        type=str,
        default=None,
        help="Root directory of CLRS PyTorch datasets (defaults to env's internal default).",
    )

    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=47)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)

    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def make_env_fn(args: argparse.Namespace):
    def _make_env():
        env = BellmanFordEnv(
            n_nodes=args.n_nodes,
            reward_mode=args.reward_mode,
            seed=None,
            use_clrs=not args.no_clrs,
            clrs_root=args.clrs_root,
        )
        return env

    return _make_env


def main():
    args = parse_args()

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # When using SubprocVecEnv, env stepping happens in worker processes.
    # Keep torch threads modest in the main learner process to avoid oversubscription.
    torch_threads = max(1, min(4, slurm_cpus))
    torch.set_num_threads(torch_threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "bf_ppo" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_fn = make_env_fn(args)

    env = make_vec_env(
        env_fn,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(run_dir / "monitor"),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )

    eval_env = make_vec_env(
        env_fn,
        n_envs=1,
        seed=args.seed + 10_000,
        monitor_dir=str(run_dir / "eval_monitor"),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=device,
        verbose=1,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_dir = run_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="bf_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save(str(run_dir / "final_model"))


if __name__ == "__main__":
    main()
