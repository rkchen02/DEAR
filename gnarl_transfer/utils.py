from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_compact_local() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def safe_git_commit() -> Optional[str]:
    """
    Best-effort: return current git commit hash if available, else None.
    """
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def slurm_cpus_per_task(default: int = 1) -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default
