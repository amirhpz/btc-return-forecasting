from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from btc_forecasting.data.canonical_5m import CANONICAL_RELATIVE_PATH, DECIMAL_TYPE

FIVE_MINUTES_MICROSECONDS = 300_000_000
HOUR_MICROSECONDS = 3_600_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
CANONICAL_1H_RELATIVE_PATH = Path("data/processed/btcusdt_1h_v001.parquet")
INCOMPLETE_HOURS_RELATIVE_PATH = Path("outputs/data/e00c2/incomplete_hours.csv")

ADDITIVE_COLUMNS = (
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
CANONICAL_1H_SCHEMA = pa.schema(
    [
        pa.field("open_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("open", DECIMAL_TYPE, nullable=False),
        pa.field("high", DECIMAL_TYPE, nullable=False),
        pa.field("low", DECIMAL_TYPE, nullable=False),
        pa.field("close", DECIMAL_TYPE, nullable=False),
        pa.field("volume", DECIMAL_TYPE, nullable=False),
        pa.field("quote_asset_volume", DECIMAL_TYPE, nullable=False),
        pa.field("number_of_trades", pa.int64(), nullable=False),
        pa.field("taker_buy_base_volume", DECIMAL_TYPE, nullable=False),
        pa.field("taker_buy_quote_volume", DECIMAL_TYPE, nullable=False),
    ]
)
INCOMPLETE_HOUR_FIELDS = (
    "open_time",
    "source_candle_count",
    "missing_timestamp_count",
    "missing_timestamps",
)


@dataclass(frozen=True)
class HourlyResampleResult:
    table: pa.Table
    incomplete_hours: list[dict[str, object]]
    total_possible_hours: int
    every_retained_hour_complete: bool


@dataclass(frozen=True)
class Canonical1hBuildResult:
    artifact_path: Path
    incomplete_hours_path: Path
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


class _HourlyAccumulator:
    def __init__(self) -> None:
        self.current_hour_us: int | None = None
        self.current_rows: list[dict[str, object]] = []
        self.complete_rows: list[dict[str, object]] = []
        self.incomplete_hours: list[dict[str, object]] = []
        self.first_hour_us: int | None = None
        self.last_hour_us: int | None = None

    def consume(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            open_time = row["open_time"]
            if not isinstance(open_time, datetime):
                raise TypeError("open_time must be an Arrow UTC timestamp")
            open_time_us = _datetime_to_microseconds(open_time)
            if open_time_us % FIVE_MINUTES_MICROSECONDS:
                raise ValueError(f"Source timestamp is off-grid: {_timestamp_text(open_time_us)}")
            hour_us = open_time_us - open_time_us % HOUR_MICROSECONDS
            if self.current_hour_us is None:
                self.current_hour_us = hour_us
                self.first_hour_us = hour_us
            elif hour_us < self.current_hour_us:
                raise ValueError("Source timestamps are not ordered")
            elif hour_us > self.current_hour_us:
                previous_hour = self.current_hour_us
                self._finalize_current_hour()
                missing_hour = previous_hour + HOUR_MICROSECONDS
                while missing_hour < hour_us:
                    self._record_incomplete_hour(missing_hour, [])
                    missing_hour += HOUR_MICROSECONDS
                self.current_hour_us = hour_us
            self.current_rows.append(row)

    def finish(self) -> HourlyResampleResult:
        if self.current_hour_us is None or self.first_hour_us is None:
            raise ValueError("Canonical 5-minute dataset is empty")
        self._finalize_current_hour()
        assert self.last_hour_us is not None
        total_possible_hours = (
            (self.last_hour_us - self.first_hour_us) // HOUR_MICROSECONDS + 1
        )
        table = pa.Table.from_pylist(self.complete_rows, schema=CANONICAL_1H_SCHEMA)
        return HourlyResampleResult(
            table=table,
            incomplete_hours=self.incomplete_hours,
            total_possible_hours=total_possible_hours,
            every_retained_hour_complete=(
                len(self.complete_rows) + len(self.incomplete_hours) == total_possible_hours
            ),
        )

    def _finalize_current_hour(self) -> None:
        assert self.current_hour_us is not None
        self.last_hour_us = self.current_hour_us
        actual_timestamps = [
            _datetime_to_microseconds(row["open_time"])  # type: ignore[arg-type]
            for row in self.current_rows
        ]
        expected_timestamps = [
            self.current_hour_us + index * FIVE_MINUTES_MICROSECONDS for index in range(12)
        ]
        if actual_timestamps == expected_timestamps:
            self.complete_rows.append(self._aggregate_current_hour())
        else:
            self._record_incomplete_hour(self.current_hour_us, actual_timestamps)
        self.current_rows = []

    def _aggregate_current_hour(self) -> dict[str, object]:
        first = self.current_rows[0]
        last = self.current_rows[-1]
        return {
            "open_time": self.current_hour_us,
            "open": first["open"],
            "high": max(row["high"] for row in self.current_rows),
            "low": min(row["low"] for row in self.current_rows),
            "close": last["close"],
            "volume": sum(
                (row["volume"] for row in self.current_rows), start=Decimal(0)
            ),
            "quote_asset_volume": sum(
                (row["quote_asset_volume"] for row in self.current_rows),
                start=Decimal(0),
            ),
            "number_of_trades": sum(
                int(row["number_of_trades"]) for row in self.current_rows
            ),
            "taker_buy_base_volume": sum(
                (row["taker_buy_base_volume"] for row in self.current_rows),
                start=Decimal(0),
            ),
            "taker_buy_quote_volume": sum(
                (row["taker_buy_quote_volume"] for row in self.current_rows),
                start=Decimal(0),
            ),
        }

    def _record_incomplete_hour(
        self,
        hour_us: int,
        actual_timestamps: list[int],
    ) -> None:
        actual_set = set(actual_timestamps)
        expected_timestamps = [
            hour_us + index * FIVE_MINUTES_MICROSECONDS for index in range(12)
        ]
        missing_timestamps = [
            timestamp for timestamp in expected_timestamps if timestamp not in actual_set
        ]
        self.incomplete_hours.append(
            {
                "open_time": _timestamp_text(hour_us),
                "source_candle_count": len(actual_timestamps),
                "missing_timestamp_count": len(missing_timestamps),
                "missing_timestamps": ";".join(
                    _timestamp_text(timestamp) for timestamp in missing_timestamps
                ),
            }
        )
        self.last_hour_us = hour_us


def resample_complete_hours(source: pa.Table) -> HourlyResampleResult:
    """Retain only UTC hours containing the exact twelve expected 5-minute timestamps."""
    accumulator = _HourlyAccumulator()
    accumulator.consume(source.to_pylist())
    return accumulator.finish()


def _verify_hourly_table(
    path: Path,
    *,
    resample_result: HourlyResampleResult,
) -> dict[str, object]:
    table = pq.read_table(path, columns=["open_time"])
    open_times = np.asarray(
        table.column("open_time")
        .combine_chunks()
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if len(open_times) == 0:
        raise ValueError("Canonical 1-hour dataset is empty")
    return {
        "total_possible_utc_hourly_intervals": resample_result.total_possible_hours,
        "complete_1h_bars_produced": int(len(open_times)),
        "incomplete_hours_excluded": len(resample_result.incomplete_hours),
        "first_1h_timestamp": _timestamp_text(int(open_times[0])),
        "last_1h_timestamp": _timestamp_text(int(open_times[-1])),
        "duplicate_timestamp_count": int(len(open_times) - np.unique(open_times).size),
        "off_grid_timestamp_count": int(
            np.count_nonzero(open_times % HOUR_MICROSECONDS)
        ),
        "every_retained_hour_had_exactly_12_expected_source_candles": (
            resample_result.every_retained_hour_complete
            and len(open_times) + len(resample_result.incomplete_hours)
            == resample_result.total_possible_hours
        ),
    }


def _write_incomplete_hours(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INCOMPLETE_HOUR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_canonical_1h_dataset(
    *,
    source_path: Path,
    output_path: Path,
    incomplete_hours_path: Path,
) -> Canonical1hBuildResult:
    """Build and verify canonical complete UTC hours without changing the 5-minute input."""
    source_signature_before = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    accumulator = _HourlyAccumulator()
    source = pq.ParquetFile(source_path)
    for batch in source.iter_batches():
        accumulator.consume(batch.to_pylist())
    resample_result = accumulator.finish()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_hours_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    incomplete_temporary_path = incomplete_hours_path.with_suffix(
        f"{incomplete_hours_path.suffix}.tmp"
    )
    pq.write_table(
        resample_result.table,
        output_temporary_path,
        compression="snappy",
    )
    _write_incomplete_hours(incomplete_temporary_path, resample_result.incomplete_hours)
    verification = _verify_hourly_table(
        output_temporary_path,
        resample_result=resample_result,
    )
    verification["output_sha256"] = _sha256(output_temporary_path)

    required_values = {
        "duplicate_timestamp_count": 0,
        "off_grid_timestamp_count": 0,
        "every_retained_hour_had_exactly_12_expected_source_candles": True,
    }
    failures = {
        key: (verification[key], expected)
        for key, expected in required_values.items()
        if verification[key] != expected
    }
    if failures:
        raise ValueError(f"Canonical 1-hour dataset verification failed: {failures}")
    source_signature_after = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    if source_signature_after != source_signature_before:
        raise ValueError("Canonical 5-minute source was modified during resampling")

    os.replace(output_temporary_path, output_path)
    os.replace(incomplete_temporary_path, incomplete_hours_path)
    return Canonical1hBuildResult(
        artifact_path=output_path,
        incomplete_hours_path=incomplete_hours_path,
        verification=verification,
    )


def run_canonical_1h_build(*, project_root: Path) -> Canonical1hBuildResult:
    root = project_root.resolve()
    return build_canonical_1h_dataset(
        source_path=root / CANONICAL_RELATIVE_PATH,
        output_path=root / CANONICAL_1H_RELATIVE_PATH,
        incomplete_hours_path=root / INCOMPLETE_HOURS_RELATIVE_PATH,
    )
