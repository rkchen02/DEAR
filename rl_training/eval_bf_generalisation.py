#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.bellman_ford_env import BellmanFordEnv


def _resolve_model_path(run_id: str, which: str) -> Path:
    run_dir = Path("runs") / "bf_ppo" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    if which == "final":
        p = run_dir / "final_model.zip"
        if not p.exists():
            raise FileNotFoundError(f"final model not found: {p}")
        return p

    if which == "best":
        p = run_dir / "best_model" / "best_model.zip"
        if not p.exists():
            raise FileNotFoundError(f"best model not found: {p}")
        return p

    if which == "checkpoint":
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
        # pick latest by timestep in filename bf_ppo_<steps>_steps.zip
        ckpts = sorted(ckpt_dir.glob("bf_ppo_*_steps.zip"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints found in: {ckpt_dir}")
        return ckpts[-1]

    raise ValueError(f"unknown which={which!r}")


def _make_fixed_env(
    *,
    fixed_nodes: int,
    max_nodes: int,
    seed: int,
    clrs_root: Optional[str],
    clrs_split: str,
) -> Any:
    def _thunk():
        return BellmanFordEnv(
            n_nodes=fixed_nodes,
            max_nodes=max_nodes,
            fixed_nodes=fixed_nodes,
            train_nodes=[fixed_nodes],
            reward_mode="sparse",  # reward doesn’t matter for pure eval; keep consistent
            seed=None,
            use_clrs=True,
            clrs_root=clrs_root,
            clrs_split=clrs_split,
            clrs_num_nodes_list=[fixed_nodes],
        )
    return _thunk


@dataclass
class Agg:
    success: List[float]
    dist_error: List[float]
    pred_acc: List[float]

    def add(self, info: Dict[str, Any]) -> None:
        self.success.append(float(info.get("is_success", 0.0)))
        self.dist_error.append(float(info.get("dist_error", np.nan)))
        self.pred_acc.append(float(info.get("pred_accuracy", np.nan)))

    def summary(self) -> Dict[str, float]:
        def mean_std(xs: List[float]) -> tuple[float, float]:
            arr = np.asarray(xs, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            return float(arr.mean()), float(arr.std(ddof=0))

        s_m, s_sd = mean_std(self.success)
        d_m, d_sd = mean_std(self.dist_error)
        p_m, p_sd = mean_std(self.pred_acc)
        return {
            "success_rate": s_m,
            "success_rate_std": s_sd,
            "dist_error": d_m,
            "dist_error_std": d_sd,
            "pred_accuracy": p_m,
            "pred_accuracy_std": p_sd,
        }


def eval_one_size(
    *,
    model: PPO,
    fixed_nodes: int,
    max_nodes: int,
    episodes: int,
    n_envs: int,
    seed: int,
    clrs_root: Optional[str],
    clrs_split: str,
) -> Dict[str, float]:
    env_fn = _make_fixed_env(
        fixed_nodes=fixed_nodes,
        max_nodes=max_nodes,
        seed=seed,
        clrs_root=clrs_root,
        clrs_split=clrs_split,
    )
    vec = make_vec_env(
        env_fn,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        vec_env_kwargs={"start_method": "forkserver"} if n_envs > 1 else None,
    )

    agg = Agg(success=[], dist_error=[], pred_acc=[])
    ep_done = 0

    obs = vec.reset()
    while ep_done < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec.step(action)
        for done, info in zip(dones, infos):
            if not done:
                continue
            # Expect env to attach metrics in info at termination.
            agg.add(info)
            ep_done += 1
            if ep_done >= episodes:
                break

    vec.close()
    return agg.summary()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None, help="runs/bf_ppo/<run-id>/...")
    ap.add_argument("--which", type=str, choices=["best", "final", "checkpoint"], default="best")
    ap.add_argument("--model-path", type=str, default=None, help="Direct path to .zip model")
    ap.add_argument("--clrs-root", type=str, default=None)
    ap.add_argument("--clrs-split", type=str, default="val")
    ap.add_argument("--max-nodes", type=int, default=64)
    ap.add_argument("--sizes", type=str, default="16,64")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if not args.run_id:
            raise SystemExit("Need either --model-path or --run-id")
        model_path = _resolve_model_path(args.run_id, args.which)

    # CPU-only is fine (and avoids GPU contention). Keep threads modest.
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(max(1, min(4, slurm_cpus)))
    device = "cpu"

    model = PPO.load(str(model_path), device=device)

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    for n in sizes:
        res = eval_one_size(
            model=model,
            fixed_nodes=n,
            max_nodes=args.max_nodes,
            episodes=args.episodes,
            n_envs=args.n_envs,
            seed=args.seed + n * 1000,
            clrs_root=args.clrs_root,
            clrs_split=args.clrs_split,
        )
        results[str(n)] = res
        print(f"[eval] n={n} -> {res}")

    # write json + csv
    (out_dir / "generalisation_results.json").write_text(json.dumps(results, indent=2))

    csv_lines = ["n,success_rate,success_rate_std,dist_error,dist_error_std,pred_accuracy,pred_accuracy_std"]
    for n_str, r in results.items():
        csv_lines.append(
            ",".join(
                [
                    n_str,
                    str(r.get("success_rate", "")),
                    str(r.get("success_rate_std", "")),
                    str(r.get("dist_error", "")),
                    str(r.get("dist_error_std", "")),
                    str(r.get("pred_accuracy", "")),
                    str(r.get("pred_accuracy_std", "")),
                ]
            )
        )
    (out_dir / "generalisation_results.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"[eval] wrote {out_dir/'generalisation_results.json'} and CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
