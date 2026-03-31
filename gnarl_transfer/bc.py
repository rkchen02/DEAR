from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .expert_data import ExpertDataset

OBS_KEYS: Tuple[str, ...] = ("d", "pred", "visited", "source", "t", "p")


@dataclass(frozen=True)
class BCConfig:
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5
    num_workers: int = 0
    seed: int = 47


@dataclass(frozen=True)
class BCHistory:
    train_loss: List[float]
    val_loss: List[float]
    val_accuracy: List[float]


class BellmanFordImitationDataset(Dataset):
    def __init__(
        self,
        dataset: ExpertDataset,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.dataset = dataset
        if indices is None:
            self.indices = np.arange(len(dataset), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        i = int(self.indices[idx])

        obs: Dict[str, torch.Tensor] = {
            "d": torch.as_tensor(self.dataset.observations["d"][i], dtype=torch.float32),
            "pred": torch.as_tensor(self.dataset.observations["pred"][i], dtype=torch.float32),
            "visited": torch.as_tensor(self.dataset.observations["visited"][i], dtype=torch.float32),
            "source": torch.as_tensor(np.asarray(self.dataset.observations["source"][i]).reshape(1), dtype=torch.float32),
            "t": torch.as_tensor(np.asarray(self.dataset.observations["t"][i]).reshape(1), dtype=torch.float32),
            "p": torch.as_tensor(np.asarray(self.dataset.observations["p"][i]).reshape(1), dtype=torch.float32),
        }
        action = torch.as_tensor(self.dataset.actions[i], dtype=torch.long)
        return obs, action


class BellmanFordBCPolicy(nn.Module):
    def __init__(self, max_nodes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.hidden_dim = int(hidden_dim)

        input_dim = (3 * self.max_nodes) + 3
        output_dim = self.max_nodes * self.max_nodes

        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def _flatten_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        d = obs["d"].float()
        pred = obs["pred"].float()
        visited = obs["visited"].float()
        source = obs["source"].float()
        t = obs["t"].float()
        p = obs["p"].float()

        if d.ndim == 1:
            d = d.unsqueeze(0)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        if visited.ndim == 1:
            visited = visited.unsqueeze(0)
        if source.ndim == 1:
            source = source.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if p.ndim == 1:
            p = p.unsqueeze(-1)

        return torch.cat([d, pred, visited, source, t, p], dim=-1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._flatten_obs(obs)
        return self.net(x)

    def predict_action(self, obs: Dict[str, Any], device: str | torch.device = "cpu") -> int:
        batch = {
            "d": torch.as_tensor(np.asarray(obs["d"]), dtype=torch.float32, device=device).unsqueeze(0),
            "pred": torch.as_tensor(np.asarray(obs["pred"]), dtype=torch.float32, device=device).unsqueeze(0),
            "visited": torch.as_tensor(np.asarray(obs["visited"]), dtype=torch.float32, device=device).unsqueeze(0),
            "source": torch.as_tensor(np.asarray(obs["source"]).reshape(1, 1), dtype=torch.float32, device=device),
            "t": torch.as_tensor(np.asarray(obs["t"]).reshape(1, 1), dtype=torch.float32, device=device),
            "p": torch.as_tensor(np.asarray(obs["p"]).reshape(1, 1), dtype=torch.float32, device=device),
        }

        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            return int(torch.argmax(logits, dim=-1).item())


def split_indices(
    n_samples: int,
    train_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be strictly between 0 and 1")

    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)

    split = max(1, min(n_samples - 1, int(round(train_fraction * n_samples))))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return train_idx, val_idx


def _move_obs_to_device(
    obs: Dict[str, torch.Tensor],
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in obs.items()}


def supervised_accuracy(
    model: BellmanFordBCPolicy,
    dataloader: DataLoader,
    device: str | torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for obs, actions in dataloader:
            obs = _move_obs_to_device(obs, device)
            actions = actions.to(device)

            logits = model(obs)
            loss = F.cross_entropy(logits, actions)

            loss_sum += float(loss.item()) * int(actions.shape[0])
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == actions).sum().item())
            total += int(actions.shape[0])

    if total == 0:
        return 0.0, 0.0

    return loss_sum / total, correct / total


def train_bc_model(
    train_dataset: BellmanFordImitationDataset,
    val_dataset: BellmanFordImitationDataset,
    *,
    max_nodes: int,
    config: BCConfig,
    device: str | torch.device = "cpu",
) -> Tuple[BellmanFordBCPolicy, BCHistory]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = BellmanFordBCPolicy(max_nodes=max_nodes, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = config.patience

    for _ in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0

        for obs, actions in train_loader:
            obs = _move_obs_to_device(obs, device)
            actions = actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(obs)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()

            batch_size = int(actions.shape[0])
            running_loss += float(loss.item()) * batch_size
            running_count += batch_size

        epoch_train_loss = running_loss / max(1, running_count)
        epoch_val_loss, epoch_val_accuracy = supervised_accuracy(model, val_loader, device)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = BCHistory(
        train_loss=train_losses,
        val_loss=val_losses,
        val_accuracy=val_accuracies,
    )
    return model, history


def save_bc_checkpoint(
    path: str | Path,
    model: BellmanFordBCPolicy,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "max_nodes": model.max_nodes,
        "hidden_dim": model.hidden_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_bc_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[BellmanFordBCPolicy, Dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {in_path}")

    payload = torch.load(in_path, map_location=device)
    model = BellmanFordBCPolicy(
        max_nodes=int(payload["max_nodes"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    metadata = dict(payload.get("metadata", {}))
    return model, metadata


def rollout_bc_policy(
    env_factory: Any,
    model: BellmanFordBCPolicy,
    *,
    episodes: int,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    env = env_factory()

    success: List[float] = []
    dist_error: List[float] = []
    pred_accuracy: List[float] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            done = False

            while not done:
                action = model.predict_action(obs, device=device)
                obs, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

            success.append(float(info.get("is_success", 0.0)))
            if "dist_error" in info:
                dist_error.append(float(info["dist_error"]))
            if "pred_accuracy" in info:
                pred_accuracy.append(float(info["pred_accuracy"]))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    def _mean(xs: Iterable[float]) -> float:
        arr = np.asarray(list(xs), dtype=np.float64)
        return float(arr.mean()) if arr.size > 0 else float("nan")

    def _std(xs: Iterable[float]) -> float:
        arr = np.asarray(list(xs), dtype=np.float64)
        return float(arr.std(ddof=0)) if arr.size > 0 else float("nan")

    return {
        "success_rate": _mean(success),
        "success_rate_std": _std(success),
        "dist_error": _mean(dist_error),
        "dist_error_std": _std(dist_error),
        "pred_accuracy": _mean(pred_accuracy),
        "pred_accuracy_std": _std(pred_accuracy),
    }


def save_history_json(path: str | Path, history: BCHistory) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(history), indent=2))
    return out_path