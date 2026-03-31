from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

ObsDict = Dict[str, np.ndarray]
EnvFactory = Callable[[], Any]


@dataclass(frozen=True)
class ExpertDataset:
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    episode_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def _normalise_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    normalised: Dict[str, np.ndarray] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        normalised[key] = arr.copy()
    return normalised


def _append_obs(buffers: Dict[str, List[np.ndarray]], obs: Dict[str, Any]) -> None:
    normalised = _normalise_obs(obs)
    for key, value in normalised.items():
        if key not in buffers:
            buffers[key] = []
        buffers[key].append(value)


def _stack_obs_buffers(buffers: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    stacked: Dict[str, np.ndarray] = {}
    for key, values in buffers.items():
        stacked[key] = np.stack(values, axis=0)
    return stacked


def collect_bf_expert_data(
    env_factory: EnvFactory,
    episodes: int,
    seed: int = 0,
) -> ExpertDataset:
    """
    Collect expert trajectories by executing the environment's canonical
    Bellman-Ford teacher policy.

    The environment is expected to expose:
      - get_expert_action()
      - standard Gymnasium reset() and step()
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    env = env_factory()
    obs_buffers: Dict[str, List[np.ndarray]] = {}
    actions: List[int] = []
    episode_ids: List[int] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False

            while not done:
                _append_obs(obs_buffers, obs)
                action = int(env.get_expert_action())
                actions.append(action)
                episode_ids.append(ep)

                obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return ExpertDataset(
        observations=_stack_obs_buffers(obs_buffers),
        actions=np.asarray(actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def save_expert_dataset(path: str | Path, dataset: ExpertDataset) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, np.ndarray] = {
        "actions": dataset.actions,
        "episode_ids": dataset.episode_ids,
    }
    for key, value in dataset.observations.items():
        payload[f"obs_{key}"] = value

    np.savez_compressed(out_path, **payload)
    return out_path


def load_expert_dataset(path: str | Path) -> ExpertDataset:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Expert dataset not found: {in_path}")

    with np.load(in_path, allow_pickle=False) as data:
        observations = {
            key[len("obs_"):]: data[key]
            for key in data.files
            if key.startswith("obs_")
        }
        actions = data["actions"]
        episode_ids = data["episode_ids"]

    return ExpertDataset(
        observations=observations,
        actions=actions.astype(np.int64, copy=False),
        episode_ids=episode_ids.astype(np.int64, copy=False),
    )