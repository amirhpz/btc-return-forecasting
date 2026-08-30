from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from btc_forecasting.common.hashing import sha256_file


def _git_state(root: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def create_run_manifest(
    *,
    project_root: Path,
    experiment_id: str,
    run_id: str,
    raw_data_path: Path | None = None,
) -> dict[str, Any]:
    """Create a minimal immutable run manifest."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_state(project_root),
    }
    if raw_data_path is not None and raw_data_path.is_file():
        manifest["raw_data"] = {
            "path": str(raw_data_path),
            "sha256": sha256_file(raw_data_path),
            "size_bytes": raw_data_path.stat().st_size,
        }
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
