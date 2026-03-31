#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from stable_baselines3 import PPO

from envs.bellman_ford_env import BellmanFordEnv
from gnarl_transfer.bc import load_bc_checkpoint, rollout_bc_policy
from rl_training.eval_bf_generalisation import eval_one_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPO and BC transfer models on Bellman-Ford.")
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        help="Format: kind:name:path where kind is ppo or bc",
    )
    parser.add_argument("--sizes", type=str, default="6,16")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--clrs-root", type=str, default=None)
    parser.add_argument("--clrs-split", type=str, default="val")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-nodes", type=int, default=None)
    return parser.parse_args()


def parse_model_spec(spec: str) -> Tuple[str, str, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --model-spec: {spec!r}")
    kind, name, path_str = parts
    if kind not in {"ppo", "bc"}:
        raise ValueError(f"Invalid model kind in {spec!r}")
    return kind, name, Path(path_str)


def infer_max_nodes_from_ppo(model: PPO) -> int:
    n_actions = int(model.action_space.n)
    root = int(math.isqrt(n_actions))
    if root * root != n_actions:
        raise ValueError(f"Action space size {n_actions} is not a perfect square")
    return root


def make_fixed_env(
    *,
    fixed_nodes: int,
    max_nodes: int,
    clrs_root: str | None,
    clrs_split: str,
):
    def _make_env():
        return BellmanFordEnv(
            n_nodes=fixed_nodes,
            max_nodes=max_nodes,
            train_nodes=[fixed_nodes],
            fixed_nodes=fixed_nodes,
            reward_mode="sparse",
            seed=None,
            use_clrs=True,
            clrs_root=clrs_root,
            clrs_split=clrs_split,
            clrs_num_nodes_list=[fixed_nodes],
        )
    return _make_env


def main() -> int:
    args = parse_args()
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for spec in args.model_spec:
        kind, name, path = parse_model_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        if kind == "ppo":
            model = PPO.load(str(path), device="cpu")
            max_nodes = args.max_nodes if args.max_nodes is not None else infer_max_nodes_from_ppo(model)

            for n in sizes:
                result = eval_one_size(
                    model=model,
                    fixed_nodes=n,
                    max_nodes=max_nodes,
                    episodes=args.episodes,
                    n_envs=args.n_envs,
                    seed=args.seed + n * 1000,
                    clrs_root=args.clrs_root,
                    clrs_split=args.clrs_split,
                )
                rows.append(
                    {
                        "kind": kind,
                        "name": name,
                        "path": str(path),
                        "eval_n": n,
                        **result,
                    }
                )

        else:
            model, metadata = load_bc_checkpoint(path, device="cpu")
            max_nodes = args.max_nodes if args.max_nodes is not None else int(metadata.get("max_nodes", model.max_nodes))

            for n in sizes:
                env_factory = make_fixed_env(
                    fixed_nodes=n,
                    max_nodes=max_nodes,
                    clrs_root=args.clrs_root,
                    clrs_split=args.clrs_split,
                )
                result = rollout_bc_policy(
                    env_factory,
                    model,
                    episodes=args.episodes,
                    device="cpu",
                )
                rows.append(
                    {
                        "kind": kind,
                        "name": name,
                        "path": str(path),
                        "eval_n": n,
                        **result,
                    }
                )

    (out_dir / "transfer_results.json").write_text(json.dumps(rows, indent=2))

    header = [
        "kind",
        "name",
        "path",
        "eval_n",
        "success_rate",
        "success_rate_std",
        "dist_error",
        "dist_error_std",
        "pred_accuracy",
        "pred_accuracy_std",
    ]
    csv_lines = [",".join(header)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(col, "")) for col in header))
    (out_dir / "transfer_results.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"[eval] wrote {out_dir / 'transfer_results.json'}")
    print(f"[eval] wrote {out_dir / 'transfer_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())