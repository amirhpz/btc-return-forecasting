from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.evaluation.metrics import regression_loss_metrics
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH


@dataclass(frozen=True)
class DevelopmentTargets:
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class ZeroReturnRunResult:
    metrics_path: Path
    result: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _filtered_targets(
    table: pa.Table,
    *,
    decision_start: datetime,
    decision_end: datetime,
    target_end: datetime,
) -> np.ndarray:
    timestamp_type = pa.timestamp("us", tz="UTC")
    mask = pc.and_(
        pc.and_(
            pc.greater_equal(
                table.column("decision_time"),
                pa.scalar(decision_start, type=timestamp_type),
            ),
            pc.less(
                table.column("decision_time"),
                pa.scalar(decision_end, type=timestamp_type),
            ),
        ),
        pc.less(
            table.column("target_time"),
            pa.scalar(target_end, type=timestamp_type),
        ),
    )
    selected = table.filter(mask)
    order = pc.sort_indices(selected, sort_keys=[("decision_time", "ascending")])
    selected = selected.take(order)
    return np.asarray(
        selected.column(TARGET_COLUMN).combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def load_development_targets(
    *,
    source_path: Path,
    split_metadata_path: Path,
) -> DevelopmentTargets:
    """Load only frozen train/validation targets; test rows are excluded at read time."""
    metadata = load_yaml(split_metadata_path)
    split = metadata["split"]
    boundaries = split["boundaries"]
    train_boundary = boundaries["train"]
    validation_boundary = boundaries["validation"]
    test_start = _parse_utc(
        boundaries["test"]["decision_time_start_inclusive"]
    )
    development = pq.read_table(
        source_path,
        columns=["decision_time", "target_time", TARGET_COLUMN],
        filters=[("decision_time", "<", test_start)],
    )
    train_start = _parse_utc(train_boundary["decision_time_start_inclusive"])
    validation_start = _parse_utc(
        validation_boundary["decision_time_start_inclusive"]
    )
    train = _filtered_targets(
        development,
        decision_start=train_start,
        decision_end=validation_start,
        target_end=_parse_utc(train_boundary["target_time_end_exclusive"]),
    )
    validation = _filtered_targets(
        development,
        decision_start=validation_start,
        decision_end=test_start,
        target_end=_parse_utc(validation_boundary["target_time_end_exclusive"]),
    )
    expected = split["retained_rows"]
    if len(train) != expected["train"] or len(validation) != expected["validation"]:
        raise ValueError(
            "Development split counts do not match frozen metadata: "
            f"train={len(train)}, validation={len(validation)}"
        )
    return DevelopmentTargets(train=train, validation=validation)


def evaluate_zero_return_baseline(
    *,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
) -> dict[str, object]:
    """Evaluate deterministic zero predictions on development splits only."""
    metrics: dict[str, dict[str, object]] = {}
    reference_losses: dict[str, dict[str, float]] = {}
    for split_name, targets in (
        ("train", np.asarray(train_targets, dtype=float)),
        ("validation", np.asarray(validation_targets, dtype=float)),
    ):
        predictions = zero_return_prediction(len(targets))
        split_metrics = regression_loss_metrics(targets, predictions)
        metrics[split_name] = split_metrics
        reference_losses[split_name] = {
            "zero_return_mae": float(split_metrics["mae"]),
            "zero_return_rmse": float(split_metrics["rmse"]),
        }
    return {
        "experiment_id": "E01",
        "model": "zero_return",
        "evaluated_splits": ["train", "validation"],
        "metrics": metrics,
        "reference_losses": reference_losses,
        "skill_relative_to_self": {"skill_mae": 0.0, "skill_rmse": 0.0},
        "metric_notes": {
            "pearson_ic": "N/A — constant prediction",
            "spearman_rank_ic": "N/A — constant prediction",
            "directional_accuracy": "N/A — zero prediction has no directional sign",
        },
        "test_set": "NOT EVALUATED",
    }


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def run_zero_return_baseline(*, project_root: Path) -> ZeroReturnRunResult:
    root = project_root.resolve()
    source_path = root / TARGET_RELATIVE_PATH
    split_path = root / FROZEN_SPLIT_RELATIVE_PATH
    targets = load_development_targets(
        source_path=source_path,
        split_metadata_path=split_path,
    )
    result = evaluate_zero_return_baseline(
        train_targets=targets.train,
        validation_targets=targets.validation,
    )

    run_id = f"E01_1h_F0_B0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    manifest = create_run_manifest(
        project_root=root,
        experiment_id="E01",
        run_id=run_id,
    )
    manifest.update(
        {
            "model": "zero_return",
            "timeframe": "1h",
            "horizon": "1h",
            "feature_set": "F0",
            "evaluated_splits": ["train", "validation"],
            "test_set": "NOT EVALUATED",
            "data": {
                "path": TARGET_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(source_path),
            },
            "split": {
                "path": FROZEN_SPLIT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(split_path),
            },
            "row_counts": {
                "train": len(targets.train),
                "validation": len(targets.validation),
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    metrics_path = run_directory / "metrics.json"
    _write_json(metrics_path, result)
    return ZeroReturnRunResult(metrics_path=metrics_path, result=result)
