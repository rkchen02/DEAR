from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class RunPaths:
    """
    Standardised artefact layout for a single run directory.

    runs/<exp_name>/<run_id>/
      config.json
      metadata.json
      final_model.zip
      emergency_model.zip        (optional)
      checkpoints/
      best_model/                (SB3 EvalCallback best model dir)
      monitor/
      eval_monitor/
      eval/                      (SB3 EvalCallback log_path)
    """
    root: Path

    config_file: Path = field(init=False)
    metadata_file: Path = field(init=False)

    final_model_file: Path = field(init=False)
    emergency_model_file: Path = field(init=False)

    checkpoints_dir: Path = field(init=False)
    best_model_dir: Path = field(init=False)
    monitor_dir: Path = field(init=False)
    eval_monitor_dir: Path = field(init=False)
    eval_log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_file", self.root / "config.json")
        object.__setattr__(self, "metadata_file", self.root / "metadata.json")

        object.__setattr__(self, "final_model_file", self.root / "final_model.zip")
        object.__setattr__(self, "emergency_model_file", self.root / "emergency_model.zip")

        object.__setattr__(self, "checkpoints_dir", self.root / "checkpoints")
        object.__setattr__(self, "best_model_dir", self.root / "best_model")
        object.__setattr__(self, "monitor_dir", self.root / "monitor")
        object.__setattr__(self, "eval_monitor_dir", self.root / "eval_monitor")
        object.__setattr__(self, "eval_log_dir", self.root / "eval")
