from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch


def copy_matching_state_dict(
    target_module: torch.nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[str]]:
    """
    Copy parameters from source_state_dict into target_module wherever the key
    exists and the tensor shapes match exactly.
    """
    target_state = target_module.state_dict()

    copied: List[str] = []
    skipped: List[str] = []

    for key, value in source_state_dict.items():
        if key not in target_state:
            skipped.append(key)
            continue
        if target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        target_state[key] = value.detach().clone()
        copied.append(key)

    target_module.load_state_dict(target_state, strict=False)
    return {"copied": copied, "skipped": skipped}


def warmstart_ppo_from_path(
    target_model: object,
    source_model_path: str | Path,
    *,
    device: str = "cpu",
) -> Dict[str, List[str]]:
    """
    Warm-start an SB3 PPO model by copying matching policy parameters from an
    existing PPO checkpoint.
    """
    from stable_baselines3 import PPO

    source_path = Path(source_model_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source PPO model not found: {source_path}")

    source_model = PPO.load(str(source_path), device=device)
    return copy_matching_state_dict(target_model.policy, source_model.policy.state_dict())