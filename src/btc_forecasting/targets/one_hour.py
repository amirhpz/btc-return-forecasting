from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from btc_forecasting.data.canonical_1h import (
    CANONICAL_1H_RELATIVE_PATH,
    CANONICAL_1H_SCHEMA,
    HOUR_MICROSECONDS,
)

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
TARGET_RELATIVE_PATH = Path("data/processed/btcusdt_1h_target_v001.parquet")
TARGET_COLUMN = "future_log_return_1h"
CANONICAL_VALUE_FIELDS = tuple(CANONICAL_1H_SCHEMA)[1:]
TARGET_SCHEMA = pa.schema(
    [
        pa.field("bar_open_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("decision_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("target_time", pa.timestamp("us", tz="UTC"), nullable=False),
        *CANONICAL_VALUE_FIELDS,
        pa.field(TARGET_COLUMN, pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class TargetConstructionResult:
    table: pa.Table
    input_row_count: int
    missing_next_hour_exclusion_count: int
    final_row_exclusion_count: int


@dataclass(frozen=True)
class OneHourTargetBuildResult:
    artifact_path: Path
    verification: dict[str, object]


def _datetime_to_microseconds(value: datetime) -> int:
    delta = value - EPOCH
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _timestamp_text(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def construct_one_hour_targets(source: pa.Table) -> TargetConstructionResult:
    """Construct targets only when the next row is exactly one real hour later."""
    rows = source.to_pylist()
    if not rows:
        raise ValueError("Canonical 1-hour dataset is empty")

    target_rows: list[dict[str, object]] = []
    missing_next_hour_exclusions = 0
    for current, following in zip(rows, rows[1:]):
        current_time = current["open_time"]
        following_time = following["open_time"]
        if not isinstance(current_time, datetime) or not isinstance(following_time, datetime):
            raise TypeError("open_time must be an Arrow UTC timestamp")
        current_time_us = _datetime_to_microseconds(current_time)
        following_time_us = _datetime_to_microseconds(following_time)
        decision_time_us = current_time_us + HOUR_MICROSECONDS
        if following_time_us != decision_time_us:
            missing_next_hour_exclusions += 1
            continue

        current_close = current["close"]
        future_close = following["close"]
        target = math.log(float(future_close / current_close))  # type: ignore[operator]
        target_rows.append(
            {
                "bar_open_time": current_time_us,
                "decision_time": decision_time_us,
                "target_time": current_time_us + 2 * HOUR_MICROSECONDS,
                **{
                    field.name: current[field.name]
                    for field in CANONICAL_VALUE_FIELDS
                },
                TARGET_COLUMN: target,
            }
        )

    return TargetConstructionResult(
        table=pa.Table.from_pylist(target_rows, schema=TARGET_SCHEMA),
        input_row_count=len(rows),
        missing_next_hour_exclusion_count=missing_next_hour_exclusions,
        final_row_exclusion_count=1,
    )


def _verify_target_table(
    path: Path,
    *,
    construction: TargetConstructionResult,
) -> dict[str, object]:
    table = pq.read_table(
        path,
        columns=["bar_open_time", "decision_time", "target_time", TARGET_COLUMN],
    )
    bar_times = np.asarray(
        table.column("bar_open_time")
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    decision_times = np.asarray(
        table.column("decision_time")
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    target_times = np.asarray(
        table.column("target_time")
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    targets = np.asarray(
        table.column(TARGET_COLUMN).combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    if len(targets) == 0:
        raise ValueError("Target dataset is empty")
    finite = np.isfinite(targets)
    exact_one_hour = bool(
        np.all(decision_times - bar_times == HOUR_MICROSECONDS)
        and np.all(target_times - decision_times == HOUR_MICROSECONDS)
    )
    return {
        "input_1h_rows": construction.input_row_count,
        "eligible_target_rows": int(len(targets)),
        "rows_excluded_missing_next_exact_hour": (
            construction.missing_next_hour_exclusion_count
        ),
        "final_row_exclusion": construction.final_row_exclusion_count,
        "first_decision_time": _timestamp_text(int(decision_times[0])),
        "last_decision_time": _timestamp_text(int(decision_times[-1])),
        "nan_or_inf_target_count": int(np.count_nonzero(~finite)),
        "target_min": float(np.min(targets)),
        "target_mean": float(np.mean(targets)),
        "target_median": float(np.median(targets)),
        "target_max": float(np.max(targets)),
        "every_target_spans_exactly_one_real_hour": exact_one_hour,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_one_hour_target_dataset(
    *,
    source_path: Path,
    output_path: Path,
) -> OneHourTargetBuildResult:
    """Build and verify E00D targets without modifying the canonical hourly input."""
    source_signature_before = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    source = pq.read_table(source_path)
    construction = construct_one_hour_targets(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(construction.table, temporary_path, compression="snappy")
    verification = _verify_target_table(temporary_path, construction=construction)
    verification["output_sha256"] = _sha256(temporary_path)
    required_values = {
        "nan_or_inf_target_count": 0,
        "every_target_spans_exactly_one_real_hour": True,
    }
    failures = {
        key: (verification[key], expected)
        for key, expected in required_values.items()
        if verification[key] != expected
    }
    if failures:
        raise ValueError(f"One-hour target verification failed: {failures}")

    source_signature_after = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    if source_signature_after != source_signature_before:
        raise ValueError("Canonical 1-hour source was modified during target construction")
    os.replace(temporary_path, output_path)
    return OneHourTargetBuildResult(
        artifact_path=output_path,
        verification=verification,
    )


def run_one_hour_target_build(*, project_root: Path) -> OneHourTargetBuildResult:
    root = project_root.resolve()
    return build_one_hour_target_dataset(
        source_path=root / CANONICAL_1H_RELATIVE_PATH,
        output_path=root / TARGET_RELATIVE_PATH,
    )
