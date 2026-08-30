from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from btc_forecasting.targets.one_hour import TARGET_RELATIVE_PATH

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FROZEN_SPLIT_RELATIVE_PATH = Path("configs/frozen/split_boundaries_v001.yaml")
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


@dataclass(frozen=True)
class FrozenSplitResult:
    metadata: dict[str, object]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class FrozenSplitBuildResult:
    metadata_path: Path
    metadata: dict[str, object]


def _timestamp_text(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _timestamp_values(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table.column(column)
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )


def create_frozen_chronological_split(
    source: pa.Table,
    *,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> FrozenSplitResult:
    """Allocate by decision time, then purge labels crossing period boundaries."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a positive test split")

    decision_times = _timestamp_values(source, "decision_time")
    target_times = _timestamp_values(source, "target_time")
    if len(decision_times) < 3:
        raise ValueError("At least three eligible targets are required")
    if len(decision_times) != len(target_times):
        raise ValueError("decision_time and target_time lengths differ")

    order = np.argsort(decision_times, kind="stable")
    ordered_decisions = decision_times[order]
    ordered_targets = target_times[order]
    if np.unique(ordered_decisions).size != len(ordered_decisions):
        raise ValueError("decision_time values must be unique")

    total_rows = len(order)
    provisional_train_count = int(np.floor(total_rows * train_fraction))
    provisional_validation_count = int(np.floor(total_rows * validation_fraction))
    provisional_test_count = (
        total_rows - provisional_train_count - provisional_validation_count
    )
    if min(
        provisional_train_count,
        provisional_validation_count,
        provisional_test_count,
    ) <= 0:
        raise ValueError("Fractions must produce non-empty provisional splits")

    validation_start_position = provisional_train_count
    test_start_position = provisional_train_count + provisional_validation_count
    validation_start = int(ordered_decisions[validation_start_position])
    test_start = int(ordered_decisions[test_start_position])

    provisional_train_positions = np.arange(0, validation_start_position)
    provisional_validation_positions = np.arange(
        validation_start_position,
        test_start_position,
    )
    test_positions = np.arange(test_start_position, total_rows)
    train_keep = ordered_targets[provisional_train_positions] < validation_start
    validation_keep = ordered_targets[provisional_validation_positions] < test_start
    retained_train_positions = provisional_train_positions[train_keep]
    retained_validation_positions = provisional_validation_positions[validation_keep]

    train_indices = tuple(int(index) for index in order[retained_train_positions])
    validation_indices = tuple(int(index) for index in order[retained_validation_positions])
    test_indices = tuple(int(index) for index in order[test_positions])
    purged_train = provisional_train_count - len(train_indices)
    purged_validation = provisional_validation_count - len(validation_indices)
    split_sets = [set(train_indices), set(validation_indices), set(test_indices)]
    zero_overlap = all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    )
    no_target_crosses = bool(
        np.all(ordered_targets[retained_train_positions] < validation_start)
        and np.all(ordered_targets[retained_validation_positions] < test_start)
    )

    metadata: dict[str, object] = {
        "split": {
            "id": "chronological_70_15_15_v001",
            "method": "chronological",
            "sort_column": "decision_time",
            "purge_column": "target_time",
            "fractions": {
                "train": train_fraction,
                "validation": validation_fraction,
                "test": round(1 - train_fraction - validation_fraction, 12),
            },
            "total_eligible_target_rows": total_rows,
            "provisional_rows": {
                "train": provisional_train_count,
                "validation": provisional_validation_count,
                "test": provisional_test_count,
            },
            "retained_rows": {
                "train": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
            },
            "purged_boundary_rows": {
                "train": purged_train,
                "validation": purged_validation,
                "total": purged_train + purged_validation,
            },
            "boundaries": {
                "train": {
                    "decision_time_start_inclusive": _timestamp_text(
                        int(ordered_decisions[0])
                    ),
                    "decision_time_end_exclusive": _timestamp_text(validation_start),
                    "last_retained_decision_time": _timestamp_text(
                        int(ordered_decisions[retained_train_positions[-1]])
                    ),
                    "target_time_end_exclusive": _timestamp_text(validation_start),
                },
                "validation": {
                    "decision_time_start_inclusive": _timestamp_text(validation_start),
                    "decision_time_end_exclusive": _timestamp_text(test_start),
                    "last_retained_decision_time": _timestamp_text(
                        int(ordered_decisions[retained_validation_positions[-1]])
                    ),
                    "target_time_end_exclusive": _timestamp_text(test_start),
                },
                "test": {
                    "decision_time_start_inclusive": _timestamp_text(test_start),
                    "decision_time_end_inclusive": _timestamp_text(
                        int(ordered_decisions[-1])
                    ),
                    "target_time_end_inclusive": _timestamp_text(
                        int(np.max(ordered_targets[test_positions]))
                    ),
                },
            },
            "invariants": {
                "chronologically_ordered": bool(np.all(np.diff(ordered_decisions) > 0)),
                "zero_overlap": zero_overlap,
                "no_target_crosses_split_boundary": no_target_crosses,
            },
        }
    }
    return FrozenSplitResult(
        metadata=metadata,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_chronological_split(
    *,
    source_path: Path,
    metadata_path: Path,
) -> FrozenSplitBuildResult:
    """Build the small tracked E00E boundary file without changing target data."""
    source_signature_before = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    source = pq.read_table(source_path, columns=["decision_time", "target_time"])
    result = create_frozen_chronological_split(source)
    metadata = dict(result.metadata)
    split = metadata["split"]
    assert isinstance(split, dict)
    split["source"] = {
        "path": TARGET_RELATIVE_PATH.as_posix(),
        "sha256": _sha256(source_path),
    }

    invariants = split["invariants"]
    assert isinstance(invariants, dict)
    if not all(invariants.values()):
        raise ValueError(f"Frozen split invariants failed: {invariants}")
    source_signature_after = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    if source_signature_after != source_signature_before:
        raise ValueError("Target source was modified while freezing split boundaries")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, metadata_path)
    return FrozenSplitBuildResult(metadata_path=metadata_path, metadata=metadata)


def run_frozen_chronological_split(*, project_root: Path) -> FrozenSplitBuildResult:
    root = project_root.resolve()
    return build_frozen_chronological_split(
        source_path=root / TARGET_RELATIVE_PATH,
        metadata_path=root / FROZEN_SPLIT_RELATIVE_PATH,
    )
