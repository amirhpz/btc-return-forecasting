from __future__ import annotations

import csv
import io
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from btc_forecasting.data.acquisition_config import AcquisitionConfig
from btc_forecasting.data.binance_archive import (
    MonthlyPeriod,
    archive_filename,
    generate_monthly_periods,
)

INTERVAL_MICROSECONDS = 300_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
EXPECTED_CORRECTION_COUNT = 241
EXPECTED_MISSING_GRID_POSITIONS = 1_703
CANONICAL_RELATIVE_PATH = Path("data/processed/btcusdt_5m_v001.parquet")
TIMESTAMP_ANOMALIES_RELATIVE_PATH = Path("outputs/data/e00b/timestamp_anomalies.csv")

DECIMAL_TYPE = pa.decimal128(38, 8)
CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("open_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("open", DECIMAL_TYPE, nullable=False),
        pa.field("high", DECIMAL_TYPE, nullable=False),
        pa.field("low", DECIMAL_TYPE, nullable=False),
        pa.field("close", DECIMAL_TYPE, nullable=False),
        pa.field("volume", DECIMAL_TYPE, nullable=False),
        pa.field("close_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("quote_asset_volume", DECIMAL_TYPE, nullable=False),
        pa.field("number_of_trades", pa.int64(), nullable=False),
        pa.field("taker_buy_base_volume", DECIMAL_TYPE, nullable=False),
        pa.field("taker_buy_quote_volume", DECIMAL_TYPE, nullable=False),
        pa.field("ignore", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class TimestampCorrection:
    archive_month: str
    raw_open_time: int
    corrected_open_time_us: int

    @property
    def key(self) -> tuple[str, int]:
        return self.archive_month, self.raw_open_time


@dataclass(frozen=True)
class CanonicalBuildResult:
    artifact_path: Path
    verification: dict[str, object]


def _datetime_to_microseconds(value: datetime) -> int:
    delta = value - EPOCH
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _parse_utc_microseconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Timestamp must be UTC: {value!r}")
    return _datetime_to_microseconds(parsed)


def _timestamp_text(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _source_timestamp_microseconds(raw_value: int, period: MonthlyPeriod) -> int:
    return raw_value * 1_000 if (period.year, period.month) < (2025, 1) else raw_value


def load_timestamp_corrections(
    path: Path,
    *,
    expected_count: int = EXPECTED_CORRECTION_COUNT,
) -> tuple[TimestampCorrection, ...]:
    """Load the frozen, collision-free E00B correction allow-list."""
    corrections: list[TimestampCorrection] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "OFF_GRID" not in row["anomaly_types"].split("+"):
                continue
            if row["mapping_status"] != "UNIQUE_MISSING_GRID":
                raise ValueError("Every frozen off-grid correction must map uniquely")
            if row["nearest_grid_currently_absent"] != "True":
                raise ValueError("Every frozen correction target must be absent")
            if row["mapping_would_collide_with_existing"] != "False":
                raise ValueError("A frozen correction cannot collide with an existing row")
            corrected_open_time_us = _parse_utc_microseconds(
                row["nearest_grid_timestamp_utc"]
            )
            if corrected_open_time_us % INTERVAL_MICROSECONDS:
                raise ValueError("A frozen correction target is not on the 5-minute grid")
            corrections.append(
                TimestampCorrection(
                    archive_month=row["archive_month"],
                    raw_open_time=int(row["raw_open_time"]),
                    corrected_open_time_us=corrected_open_time_us,
                )
            )
    if len(corrections) != expected_count:
        raise ValueError(
            f"Expected {expected_count} frozen timestamp corrections, got {len(corrections)}"
        )
    if len({correction.key for correction in corrections}) != len(corrections):
        raise ValueError("Frozen timestamp corrections contain duplicate source keys")
    return tuple(corrections)


def _archive_table(
    period: MonthlyPeriod,
    path: Path,
    *,
    expected_fields: Sequence[str],
    correction_map: dict[tuple[str, int], TimestampCorrection],
    used_corrections: set[tuple[str, int]],
) -> pa.Table:
    columns: list[list[object]] = [[] for _ in CANONICAL_SCHEMA]
    expected_member = f"{path.stem}.csv"
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open(expected_member, "r") as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="")
            reader = csv.reader(text_handle, strict=True)
            for row in reader:
                if tuple(row) == tuple(expected_fields):
                    continue
                if len(row) != len(CANONICAL_SCHEMA):
                    raise ValueError(f"Unexpected column count in {path.name}: {len(row)}")

                raw_open_time = int(row[0])
                source_open_time_us = _source_timestamp_microseconds(raw_open_time, period)
                correction_key = (period.key, raw_open_time)
                correction = correction_map.get(correction_key)
                if correction is not None:
                    if source_open_time_us % INTERVAL_MICROSECONDS == 0:
                        raise ValueError("A frozen correction source is already grid-aligned")
                    open_time_us = correction.corrected_open_time_us
                    used_corrections.add(correction_key)
                else:
                    open_time_us = source_open_time_us
                    if open_time_us % INTERVAL_MICROSECONDS:
                        raise ValueError(
                            f"Unapproved off-grid open_time {raw_open_time} in {period.key}"
                        )

                columns[0].append(open_time_us)
                for position in (1, 2, 3, 4, 5):
                    columns[position].append(Decimal(row[position]))
                columns[6].append(
                    _source_timestamp_microseconds(int(row[6]), period)
                )
                columns[7].append(Decimal(row[7]))
                columns[8].append(int(row[8]))
                columns[9].append(Decimal(row[9]))
                columns[10].append(Decimal(row[10]))
                columns[11].append(row[11])

    arrays = [
        pa.array(values, type=field.type)
        for values, field in zip(columns, CANONICAL_SCHEMA, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=CANONICAL_SCHEMA)


def verify_canonical_dataset(
    path: Path,
    *,
    cutoff: datetime,
) -> dict[str, object]:
    """Measure the frozen E00C1 timestamp invariants from the written Parquet file."""
    table = pq.read_table(path, columns=["open_time"])
    open_times = (
        table.column("open_time")
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False)
    )
    if len(open_times) == 0:
        raise ValueError("Canonical dataset is empty")
    open_times = np.asarray(open_times, dtype=np.int64)
    differences = np.diff(open_times)
    unique_count = int(np.unique(open_times).size)
    cutoff_us = _datetime_to_microseconds(cutoff)
    expected_grid_count = (cutoff_us - int(open_times[0])) // INTERVAL_MICROSECONDS
    return {
        "row_count": int(len(open_times)),
        "first_open_time": _timestamp_text(int(open_times[0])),
        "last_open_time": _timestamp_text(int(open_times[-1])),
        "grid_alignment_violation_count": int(
            np.count_nonzero(open_times % INTERVAL_MICROSECONDS)
        ),
        "timestamps_strictly_ordered": bool(np.all(differences > 0)),
        "duplicate_open_time_count": int(len(open_times) - unique_count),
        "missing_grid_position_count": int(expected_grid_count - unique_count),
    }


def _raw_signatures(paths: Sequence[Path]) -> dict[Path, tuple[int, int]]:
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def build_canonical_dataset(
    archives: Sequence[tuple[MonthlyPeriod, Path]],
    *,
    expected_fields: Sequence[str],
    corrections: Sequence[TimestampCorrection],
    output_path: Path,
    cutoff: datetime,
    expected_correction_count: int,
    expected_missing_grid_positions: int,
) -> CanonicalBuildResult:
    """Create and verify one canonical Parquet dataset without modifying raw archives."""
    correction_map = {correction.key: correction for correction in corrections}
    if len(correction_map) != len(corrections):
        raise ValueError("Timestamp corrections contain duplicate source keys")

    raw_paths = [path for _, path in archives]
    raw_signatures_before = _raw_signatures(raw_paths)
    used_corrections: set[tuple[str, int]] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with pq.ParquetWriter(
        temporary_path,
        CANONICAL_SCHEMA,
        compression="snappy",
    ) as writer:
        for period, path in archives:
            writer.write_table(
                _archive_table(
                    period,
                    path,
                    expected_fields=expected_fields,
                    correction_map=correction_map,
                    used_corrections=used_corrections,
                )
            )

    if used_corrections != set(correction_map):
        unused = sorted(set(correction_map) - used_corrections)
        raise ValueError(f"Frozen timestamp corrections not found in raw data: {unused[:3]}")

    verification = verify_canonical_dataset(temporary_path, cutoff=cutoff)
    verification["timestamp_correction_count"] = len(used_corrections)
    raw_signatures_after = _raw_signatures(raw_paths)
    modified_raw_archives = sum(
        raw_signatures_before[path] != raw_signatures_after[path] for path in raw_paths
    )
    verification["raw_archive_modified_count"] = modified_raw_archives

    required_values = {
        "grid_alignment_violation_count": 0,
        "timestamps_strictly_ordered": True,
        "duplicate_open_time_count": 0,
        "timestamp_correction_count": expected_correction_count,
        "missing_grid_position_count": expected_missing_grid_positions,
        "raw_archive_modified_count": 0,
    }
    failures = {
        key: (verification[key], expected)
        for key, expected in required_values.items()
        if verification[key] != expected
    }
    if failures:
        raise ValueError(f"Canonical dataset verification failed: {failures}")

    os.replace(temporary_path, output_path)
    return CanonicalBuildResult(artifact_path=output_path, verification=verification)


def run_canonical_5m_build(
    *,
    project_root: Path,
    config: AcquisitionConfig,
) -> CanonicalBuildResult:
    root = project_root.resolve()
    periods = generate_monthly_periods(config.start_date, config.end_date_exclusive)
    archives = [
        (
            period,
            root / config.destination / archive_filename(config.symbol, config.interval, period),
        )
        for period in periods
    ]
    corrections = load_timestamp_corrections(
        root / TIMESTAMP_ANOMALIES_RELATIVE_PATH,
        expected_count=EXPECTED_CORRECTION_COUNT,
    )
    cutoff = datetime.combine(config.end_date_exclusive, datetime.min.time(), tzinfo=UTC)
    return build_canonical_dataset(
        archives,
        expected_fields=config.expected_kline_fields,
        corrections=corrections,
        output_path=root / CANONICAL_RELATIVE_PATH,
        cutoff=cutoff,
        expected_correction_count=EXPECTED_CORRECTION_COUNT,
        expected_missing_grid_positions=EXPECTED_MISSING_GRID_POSITIONS,
    )
