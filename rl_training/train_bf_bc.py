#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from envs.bellman_ford_env import BellmanFordEnv
from gnarl_transfer.bc import (
    BCConfig,
    BellmanFordImitationDataset,
    load_bc_checkpoint,
    rollout_bc_policy,
    save_bc_checkpoint,
    save_history_json,
    split_indices,
    train_bc_model,
)
from gnarl_transfer.expert_data import collect_bf_expert_data, load_expert_dataset, save_expert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavioural cloning policy for Bellman-Ford.")

    parser.add_argument("--n-nodes", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--train-nodes", type=str, default="4,5,6")
    parser.add_argument("--eval-sizes", type=str, default="6,16")

    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--collect-episodes", type=int, default=500)
    parser.add_argument("--train-fraction", type=float, default=0.9)

    parser.add_argument("--no-clrs", action="store_true")
    parser.add_argument("--clrs-root", type=str, default=None)
    parser.add_argument("--clrs-train-split", type=str, default="train")
    parser.add_argument("--clrs-eval-split", type=str, default="val")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def _parse_nodes_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_env_fn(
    *,
    train_nodes: List[int],
    max_nodes: int,
    fixed_nodes: Optional[int],
    use_clrs: bool,
    clrs_root: Optional[str],
    clrs_split: str,
):
    def _make_env():
        n_nodes_arg = int(fixed_nodes) if fixed_nodes is not None else int(train_nodes[0])
        return BellmanFordEnv(
            n_nodes=n_nodes_arg,
            max_nodes=max_nodes,
            train_nodes=train_nodes,
            fixed_nodes=fixed_nodes,
            reward_mode="sparse",
            seed=None,
            use_clrs=use_clrs,
            clrs_root=clrs_root,
            clrs_split=clrs_split,
            clrs_num_nodes_list=sorted(set(train_nodes + ([] if fixed_nodes is None else [fixed_nodes]))),
        )
    return _make_env


def main() -> int:
    args = parse_args()

    train_nodes = _parse_nodes_csv(args.train_nodes)
    eval_sizes = _parse_nodes_csv(args.eval_sizes)

    if args.max_nodes is None:
        max_nodes = max(train_nodes + eval_sizes + [int(args.n_nodes)])
    else:
        max_nodes = int(args.max_nodes)

    run_id = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "bf_bc" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(max(1, min(4, slurm_cpus)))

    train_env_factory = make_env_fn(
        train_nodes=train_nodes,
        max_nodes=max_nodes,
        fixed_nodes=None,
        use_clrs=not args.no_clrs,
        clrs_root=args.clrs_root,
        clrs_split=args.clrs_train_split,
    )

    if args.dataset_path is None:
        dataset_path = run_dir / "expert_dataset.npz"
    else:
        dataset_path = Path(args.dataset_path)

    if dataset_path.exists():
        dataset = load_expert_dataset(dataset_path)
    else:
        dataset = collect_bf_expert_data(
            train_env_factory,
            episodes=args.collect_episodes,
            seed=args.seed,
        )
        save_expert_dataset(dataset_path, dataset)

    train_idx, val_idx = split_indices(
        n_samples=len(dataset),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    train_dataset = BellmanFordImitationDataset(dataset, indices=train_idx)
    val_dataset = BellmanFordImitationDataset(dataset, indices=val_idx)

    config = BCConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    model, history = train_bc_model(
        train_dataset,
        val_dataset,
        max_nodes=max_nodes,
        config=config,
        device=device,
    )

    checkpoint_path = run_dir / "bc_model.pt"
    save_bc_checkpoint(
        checkpoint_path,
        model,
        metadata={
            "seed": args.seed,
            "max_nodes": max_nodes,
            "train_nodes": train_nodes,
            "eval_sizes": eval_sizes,
            "device": device,
        },
    )
    save_history_json(run_dir / "history.json", history)

    eval_results = {}
    for n in eval_sizes:
        eval_env_factory = make_env_fn(
            train_nodes=train_nodes,
            max_nodes=max_nodes,
            fixed_nodes=n,
            use_clrs=not args.no_clrs,
            clrs_root=args.clrs_root,
            clrs_split=args.clrs_eval_split,
        )
        eval_results[str(n)] = rollout_bc_policy(
            eval_env_factory,
            model,
            episodes=200,
            device=device,
        )

    (run_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "max_nodes": max_nodes,
                "train_nodes": train_nodes,
                "eval_sizes": eval_sizes,
                "dataset_path": str(dataset_path),
                "device": device,
                "bc_config": {
                    "hidden_dim": args.hidden_dim,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )

    print(f"[bc] wrote checkpoint to {checkpoint_path}")
    print(f"[bc] wrote history to {run_dir / 'history.json'}")
    print(f"[bc] wrote eval results to {run_dir / 'eval_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())