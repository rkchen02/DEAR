from __future__ import annotations

import argparse

from gnarl_transfer import ExperimentRunner, RunnerConfig
from envs.bellman_ford_env import BellmanFordEnv
from rl_training.train_bf_agent import BellmanFordEvalCallback 


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=16)
    p.add_argument("--reward-mode", type=str, choices=["dense", "sparse"], default="sparse")
    p.add_argument("--no-clrs", action="store_true")
    p.add_argument("--clrs-root", type=str, default=None)
    p.add_argument("--clrs-train-split", type=str, default="train")
    p.add_argument("--clrs-eval-split", type=str, default="val")

    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=47)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)

    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def env_factory(cfg: RunnerConfig, split: str):
    return BellmanFordEnv(
        n_nodes=cfg.n_nodes,
        reward_mode=cfg.reward_mode,
        seed=None,
        use_clrs=cfg.use_clrs,
        clrs_root=cfg.clrs_root,
        clrs_split=split,
    )


def eval_callback_factory(eval_env, paths, eval_freq: int):
    return BellmanFordEvalCallback(
        eval_env,
        best_model_save_path=str(paths.best_model_dir),
        log_path=str(paths.eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )


def main() -> int:
    a = parse_args()
    cfg = RunnerConfig(
        n_nodes=a.n_nodes,
        reward_mode=a.reward_mode,
        use_clrs=not a.no_clrs,
        clrs_root=a.clrs_root,
        clrs_train_split=a.clrs_train_split,
        clrs_eval_split=a.clrs_eval_split,
        n_envs=a.n_envs,
        total_timesteps=a.total_timesteps,
        seed=a.seed,
        learning_rate=a.learning_rate,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        clip_range=a.clip_range,
        ent_coef=a.ent_coef,
        vf_coef=a.vf_coef,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        n_epochs=a.n_epochs,
        checkpoint_freq=a.checkpoint_freq,
        eval_freq=a.eval_freq,
        run_name=a.run_name,
    )

    runner = ExperimentRunner(
        env_factory=env_factory,
        eval_callback_factory=eval_callback_factory,
    )

    paths = runner.run(cfg)
    print(f"[gnarl_transfer] Run completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
