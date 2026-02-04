from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import signal

from .types import PathLike, RunPaths
from .utils import ensure_dir, now_compact_local, safe_git_commit, slurm_cpus_per_task, write_json
from .io import save_policy

EnvFactory = Callable[[str], Any]  # takes clrs_split -> env
EvalCallbackFactory = Callable[[Any, RunPaths, int], Any]  # (eval_env, paths, eval_freq) -> callback


@dataclass(frozen=True)
class RunnerConfig:
    # task/env
    n_nodes: int = 16
    reward_mode: str = "sparse"
    use_clrs: bool = True
    clrs_root: Optional[str] = None
    clrs_train_split: str = "train"
    clrs_eval_split: str = "val"

    # sb3 training
    n_envs: int = 8
    total_timesteps: int = 1_000_000
    seed: int = 47

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    checkpoint_freq: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 20

    run_name: Optional[str] = None
    exp_name: str = "bf_ppo"
    run_root: str = "runs"


class ExperimentRunner:
    """
    Clean API:
      runner = ExperimentRunner(env_factory=..., eval_callback_factory=...)
      paths = runner.run(cfg)
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[RunnerConfig, str], Any],
        eval_callback_factory: Optional[EvalCallbackFactory] = None,
        checkpoint_callback_factory: Optional[Callable[[RunPaths, int], Any]] = None,
    ) -> None:
        self.env_factory = env_factory
        self.eval_callback_factory = eval_callback_factory
        self.checkpoint_callback_factory = checkpoint_callback_factory

    def _make_run_dir(self, cfg: RunnerConfig) -> RunPaths:
        run_id = cfg.run_name or now_compact_local()
        root = Path(cfg.run_root) / cfg.exp_name / run_id
        ensure_dir(root)
        paths = RunPaths(root=root)

        # Ensure expected subdirs exist
        ensure_dir(paths.checkpoints_dir)
        ensure_dir(paths.best_model_dir)
        ensure_dir(paths.monitor_dir)
        ensure_dir(paths.eval_monitor_dir)
        ensure_dir(paths.eval_log_dir)

        return paths

    def _set_seeds_and_threads(self, cfg: RunnerConfig) -> str:
        # Threading: similar to your existing logic
        slurm_cpus = slurm_cpus_per_task(default=1)
        torch_threads = 1 if torch.cuda.is_available() else max(1, min(4, slurm_cpus))
        torch.set_num_threads(torch_threads)

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        return "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, cfg: RunnerConfig) -> RunPaths:
        paths = self._make_run_dir(cfg)
        device = self._set_seeds_and_threads(cfg)

        # Persist config + metadata early (helps even if job is preempted)
        write_json(paths.config_file, asdict(cfg))
        write_json(
            paths.metadata_file,
            {
                "format": "gnarl_transfer_run_v1",
                "algo": "PPO",
                "policy": "MultiInputPolicy",
                "git_commit": safe_git_commit(),
            },
        )

        # Lazy imports to keep module importable without SB3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv

        # Build vec envs (matches your current setup)
        env = make_vec_env(
            lambda: self.env_factory(cfg, cfg.clrs_train_split),
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            monitor_dir=str(paths.monitor_dir),
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "forkserver"},
        )

        eval_env = make_vec_env(
            lambda: self.env_factory(cfg, cfg.clrs_eval_split),
            n_envs=1,
            seed=cfg.seed + 10_000,
            monitor_dir=str(paths.eval_monitor_dir),
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "forkserver"},
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            device=device,
            verbose=1,
        )

        # Default checkpoint callback, unless user provides a factory
        if self.checkpoint_callback_factory is not None:
            checkpoint_cb = self.checkpoint_callback_factory(paths, cfg.checkpoint_freq)
        else:
            checkpoint_cb = CheckpointCallback(
                save_freq=cfg.checkpoint_freq,
                save_path=str(paths.checkpoints_dir),
                name_prefix="bf_ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )

        callbacks = [checkpoint_cb]

        # Eval callback (use your custom one if provided)
        if self.eval_callback_factory is not None:
            eval_cb = self.eval_callback_factory(eval_env, paths, cfg.eval_freq)
        else:
            from stable_baselines3.common.callbacks import EvalCallback

            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=str(paths.best_model_dir),
                log_path=str(paths.eval_log_dir),
                eval_freq=cfg.eval_freq,
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
                render=False,
            )

        callbacks.append(eval_cb)

        # Signal handling (as in your script)
        def _handle_termination(signum, frame):
            print(f"[signal] received {signum}; saving emergency checkpoint...")
            try:
                model.save(str(paths.root / "emergency_model"))
                # normalise naming
                em = paths.root / "emergency_model.zip"
                if not em.exists():
                    alt = paths.root / "emergency_model"
                    if alt.exists():
                        alt.replace(em)
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
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
        )

        # Save final model via library I/O
        save_policy(
            paths.root,
            model,
            metadata={
                "format": "gnarl_transfer_run_v1",
                "algo": "PPO",
                "policy": "MultiInputPolicy",
                "device": device,
                "seed": cfg.seed,
                "n_envs": cfg.n_envs,
                "total_timesteps": cfg.total_timesteps,
                "git_commit": safe_git_commit(),
            },
        )

        env.close()
        eval_env.close()
        return paths