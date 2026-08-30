from __future__ import annotations

from pathlib import Path
from typing import Any

from btc_forecasting.common.config import load_yaml


def load_experiment_registry(project_root: Path) -> dict[str, dict[str, Any]]:
    """Load all experiment YAML files keyed by experiment ID."""
    registry: dict[str, dict[str, Any]] = {}
    for path in sorted((project_root / "configs" / "experiments").glob("*.yaml")):
        document = load_yaml(path)
        experiment = document.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError(f"Missing experiment mapping: {path}")
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(f"Missing experiment id: {path}")
        if experiment_id in registry:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        registry[experiment_id] = experiment
    return registry
