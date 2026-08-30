from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def list_yaml_files(root: Path) -> list[Path]:
    """Return all project YAML files in deterministic order."""
    return sorted((root / "configs").rglob("*.yaml"))
