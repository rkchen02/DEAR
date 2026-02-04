from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .types import PathLike, RunPaths
from .utils import ensure_dir, read_json, write_json


def save_policy(
    run_dir: PathLike,
    model: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite_metadata: bool = True,
) -> RunPaths:
    """
    Save the final policy for a run into <run_dir>/final_model.zip
    and write <run_dir>/metadata.json.

    Note: SB3's .save() will write a .zip when given a path without ".zip";
    to avoid ambiguity, we always write final_model.zip explicitly.
    """
    root = Path(run_dir)
    ensure_dir(root)
    paths = RunPaths(root=root)

    # Save final model
    model.save(str(paths.final_model_file.with_suffix("").as_posix()))  # SB3 appends .zip
    # Ensure expected filename exists
    if not paths.final_model_file.exists():
        # In case SB3 saved without .zip depending on version, handle gracefully
        alt = paths.root / "final_model"
        if alt.exists():
            alt.replace(paths.final_model_file)

    # Save metadata
    if metadata is not None:
        if paths.metadata_file.exists() and not overwrite_metadata:
            raise FileExistsError(f"metadata.json already exists at {paths.metadata_file}")
        write_json(paths.metadata_file, metadata)

    return paths


def load_policy(
    run_dir: PathLike,
    *,
    env: Any = None,
    device: str = "auto",
    algo: str = "PPO",
) -> Any:
    """
    Load a saved policy from <run_dir>/final_model.zip.
    If metadata.json exists, you can use it externally to reconstruct env config.

    Returns: SB3 model (e.g., PPO instance).
    """
    root = Path(run_dir)
    paths = RunPaths(root=root)

    model_path = paths.final_model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if algo != "PPO":
        raise ValueError(f"Unsupported algo: {algo}. Extend load_policy() if needed.")

    from stable_baselines3 import PPO

    model = PPO.load(
        str(model_path),
        env=env,
        device=device,
        print_system_info=False,
    )
    return model


def read_metadata(run_dir: PathLike) -> Dict[str, Any]:
    root = Path(run_dir)
    paths = RunPaths(root=root)
    if not paths.metadata_file.exists():
        return {}
    return read_json(paths.metadata_file)
