import argparse
import time
from pathlib import Path
import os
import signal

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.bellman_ford_env import BellmanFordEnv


class BellmanFordEvalCallback(EvalCallback):
    """
    Extends SB3 EvalCallback to aggregate extra per-episode metrics from `info`
    during evaluation. Metrics are read when an eval episode terminates.
    Expect env to put these keys into `info` on done:
      - "dist_error" (float)
      - "pred_accuracy" (float)
      - "is_success" (bool/int)  # already used by SB3
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist_error_buffer = []
        self._pred_acc_buffer = []

    def _log_success_callback(self, locals_, globals_) -> None:
        """
        Called by SB3's evaluate_policy() after each step during evaluation.
        We hook into it to also collect our custom metrics.
        """
        # Keep SB3's default success tracking behavior.
        super()._log_success_callback(locals_, globals_)

        done = locals_.get("done")
        info = locals_.get("info")

        if not done or info is None:
            return

        # Only collect at episode end.
        if "dist_error" in info:
            try:
                self._dist_error_buffer.append(float(info["dist_error"]))
            except Exception:
                pass

        if "pred_accuracy" in info:
            try:
                self._pred_acc_buffer.append(float(info["pred_accuracy"]))
            except Exception:
                pass

    def _on_step(self) -> bool:
        # Reset buffers right before an evaluation happens (so they don't mix across evals).
        # EvalCallback triggers evaluation inside super()._on_step().
        self._dist_error_buffer = []
        self._pred_acc_buffer = []

        result = super()._on_step()

        # If an evaluation just happened, EvalCallback sets `self.last_mean_reward`.
        # We can also log our aggregated means if buffers are non-empty.
        if self.n_calls % self.eval_freq == 0:
            if len(self._dist_error_buffer) > 0:
                self.logger.record("eval/dist_error", float(np.mean(self._dist_error_buffer)))
                self.logger.record("eval/dist_error_std", float(np.std(self._dist_error_buffer)))

            if len(self._pred_acc_buffer) > 0:
                self.logger.record("eval/pred_accuracy", float(np.mean(self._pred_acc_buffer)))
                self.logger.record("eval/pred_accuracy_std", float(np.std(self._pred_acc_buffer)))

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Bellman-FordEnv.")
    parser.add_argument("--n-nodes", type=int, default=16)
    parser.add_argument("--reward-mode", type=str, choices=["dense", "sparse"], default="sparse")

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Fixed max_nodes for observation/action spaces (e.g. 64). If None, uses max(train/eval).",
    )
    parser.add_argument(
        "--train-nodes",
        type=str,
        default=None,
        help="Comma-separated node sizes to train on (e.g. '16' or '10,15'). Defaults to --n-nodes.",
    )
    parser.add_argument(
        "--eval-nodes",
        type=int,
        default=None,
        help="Fixed node size for eval env during training (defaults to --n-nodes).",
    )

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

    parser.add_argument(
        "--clrs-train-split",
        type=str,
        default="train",
        help="CLRS split for training env (train/val/test).",
    )
    parser.add_argument(
        "--clrs-eval-split",
        type=str,
        default="val",
        help="CLRS split for eval env (train/val/test).",
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


def _parse_nodes_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def make_env_fn(args: argparse.Namespace, *, split: str, fixed_nodes: int, train_nodes: list[int], max_nodes: int):
    def _make_env():
        env = BellmanFordEnv(
            n_nodes=fixed_nodes,
            max_nodes=max_nodes,
            train_nodes=train_nodes,
            fixed_nodes=fixed_nodes,
            reward_mode=args.reward_mode,
            seed=None,
            use_clrs=not args.no_clrs,
            clrs_root=args.clrs_root,
            clrs_split=split,
            clrs_num_nodes_list=sorted(set(train_nodes + [fixed_nodes])),
        )
        return env
    return _make_env


def main():
    args = parse_args()
    train_nodes = _parse_nodes_csv(args.train_nodes) if args.train_nodes else [int(args.n_nodes)]
    eval_nodes = int(args.eval_nodes) if args.eval_nodes is not None else int(args.n_nodes)
    if args.max_nodes is None:
        max_nodes = max(train_nodes + [eval_nodes])
    else:
        max_nodes = int(args.max_nodes)

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # When using SubprocVecEnv, env stepping happens in worker processes.
    # Keep torch threads modest in the main learner process to avoid oversubscription.
    torch_threads = 1 if torch.cuda.is_available() else max(1, min(4, slurm_cpus))
    torch.set_num_threads(torch_threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "bf_ppo" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_fn = make_env_fn(
        args,
        split=args.clrs_train_split,
        fixed_nodes=train_nodes[0],
        train_nodes=train_nodes,
        max_nodes=max_nodes,)

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

    eval_callback = BellmanFordEvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    def _handle_termination(signum, frame):
        print(f"[signal] received {signum}; saving emergency checkpoint...")
        try:
            model.save(str(run_dir / "emergency_model"))
            print("[signal] emergency checkpoint saved")
        except Exception as e:
            print(f"[signal] failed to save emergency checkpoint: {e!r}")
        try:
            env.close()
            eval_env.close()
        except Exception:
            pass
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGUSR1, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save(str(run_dir / "final_model"))


if __name__ == "__main__":
    main()
