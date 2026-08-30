from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from statistics import median

from btc_forecasting.data.acquisition_config import AcquisitionConfig
from btc_forecasting.data.binance_archive import (
    MonthlyPeriod,
    archive_filename,
    generate_monthly_periods,
)

INTERVAL_MICROSECONDS = 300_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
UNIT_BOUNDARY = date(2025, 1, 1)
NUMERIC_FIELD_POSITIONS = (
    ("open", 1),
    ("high", 2),
    ("low", 3),
    ("close", 4),
    ("volume", 5),
    ("quote_asset_volume", 7),
    ("number_of_trades", 8),
    ("taker_buy_base_volume", 9),
    ("taker_buy_quote_volume", 10),
    ("ignore", 11),
)
PRICE_FIELDS = ("open", "high", "low", "close")
VOLUME_FIELDS = (
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
TIMESTAMP_ANOMALY_FIELDS = (
    "archive_month",
    "raw_open_time",
    "timestamp_unit",
    "normalized_open_time_utc",
    "anomaly_types",
    "nearest_grid_timestamp_utc",
    "alternate_nearest_grid_timestamp_utc",
    "signed_grid_offset_source_units",
    "signed_grid_offset_microseconds",
    "absolute_grid_offset_microseconds",
    "nearest_grid_currently_absent",
    "mapping_would_collide_with_existing",
    "mapping_status",
    "actual_close_time_raw",
    "expected_close_time_raw",
    "signed_close_time_error_source_units",
    "signed_close_time_error_microseconds",
    "zero_volume",
    "ignore_value",
)


@dataclass
class ArchiveValidation:
    archive_month: str
    archive_filename: str
    expected_csv_member: str
    expected_timestamp_unit: str
    archive_open_status: str = "NOT_OPENED"
    archive_error: str | None = None
    zip_member_count: int = 0
    expected_payload_exists: bool = False
    unexpected_zip_member_count: int = 0
    unexpected_zip_members: str = "[]"
    row_count: int = 0
    first_open_time_utc: str | None = None
    last_open_time_utc: str | None = None
    malformed_row_count: int = 0
    column_count_problem_count: int = 0
    unexpected_header_row_count: int = 0
    timestamp_parse_failure_count: int = 0
    timestamp_unit_mismatch_count: int = 0
    alignment_violation_count: int = 0
    alignment_violation_open_times: list[str] = field(default_factory=list)
    close_time_violation_count: int = 0
    close_time_violation_details: list[dict[str, int]] = field(default_factory=list)
    ordering_violation_count: int = 0
    month_containment_violation_count: int = 0
    month_containment_violation_open_times: list[str] = field(default_factory=list)
    numeric_parse_failure_count: int = 0
    nan_count: int = 0
    infinity_count: int = 0
    invalid_numeric_row_count: int = 0
    nonpositive_price_row_count: int = 0
    ohlc_violation_count: int = 0
    negative_volume_row_count: int = 0
    zero_volume_row_count: int = 0
    negative_trade_count_row_count: int = 0
    taker_buy_base_exceeds_volume_count: int = 0
    taker_buy_quote_exceeds_quote_volume_count: int = 0
    schema_status: str = "INVALID"
    parse_status: str = "INVALID"
    validation_status: str = "INVALID"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in (
            "alignment_violation_open_times",
            "close_time_violation_details",
            "month_containment_violation_open_times",
        ):
            result[key] = json.dumps(result[key], separators=(",", ":"))
        return result


@dataclass(frozen=True)
class RawDatasetValidation:
    summary: dict[str, object]
    archive_rows: list[dict[str, object]]
    gaps: list[dict[str, object]]
    timestamp_anomalies: list[dict[str, object]]


@dataclass(frozen=True)
class RawValidationRunResult:
    exit_code: int
    summary: dict[str, object]
    summary_path: Path
    archive_validation_path: Path
    gaps_path: Path
    timestamp_anomalies_path: Path


@dataclass(frozen=True)
class _ArchiveOutcome:
    report: ArchiveValidation
    minimum_timestamp_us: int | None
    maximum_timestamp_us: int | None


@dataclass(frozen=True)
class _TimestampObservation:
    archive_month: str
    raw_open_time: int
    timestamp_unit: str
    normalized_open_time_us: int
    is_off_grid: bool
    actual_close_time_raw: int
    expected_close_time_raw: int
    is_close_time_violation: bool
    zero_volume: bool
    ignore_value: str


@dataclass
class _GlobalState:
    seen_rows: dict[int, tuple[bytes, str]]
    ignore_counts: Counter[str]
    last_timestamp_us: int | None = None
    global_ordering_violation_count: int = 0
    duplicate_timestamp_count: int = 0
    exact_duplicate_row_count: int = 0
    conflicting_duplicate_row_count: int = 0
    cross_month_duplicate_timestamp_count: int = 0
    timestamp_anomalies: list[_TimestampObservation] = field(default_factory=list)


def _expected_unit(period: MonthlyPeriod) -> str:
    return "milliseconds" if date(period.year, period.month, 1) < UNIT_BOUNDARY else "microseconds"


def _detect_unit(value: int) -> str | None:
    magnitude = abs(value)
    if 100_000_000_000 <= magnitude < 100_000_000_000_000:
        return "milliseconds"
    if 100_000_000_000_000 <= magnitude < 100_000_000_000_000_000:
        return "microseconds"
    return None


def _normalize_to_microseconds(value: int, unit: str) -> int:
    return value * 1_000 if unit == "milliseconds" else value


def _utc_datetime(timestamp_us: int) -> datetime:
    return EPOCH + timedelta(microseconds=timestamp_us)


def _timestamp_text(timestamp_us: int) -> str:
    return _utc_datetime(timestamp_us).isoformat().replace("+00:00", "Z")


def _row_digest(row: Sequence[str]) -> bytes:
    return hashlib.sha256("\x1f".join(row).encode("utf-8")).digest()


def _decimal(value: str) -> tuple[Decimal | None, str | None]:
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None, "parse"
    if result.is_nan():
        return None, "nan"
    if result.is_infinite():
        return None, "infinity"
    return result, None


def _record_numeric_validation(
    row: Sequence[str],
    report: ArchiveValidation,
    state: _GlobalState,
) -> None:
    values: dict[str, Decimal] = {}
    invalid_row = False
    for numeric_field, position in NUMERIC_FIELD_POSITIONS:
        raw_value = row[position]
        value, error = _decimal(raw_value)
        if error == "parse":
            report.numeric_parse_failure_count += 1
            invalid_row = True
        elif error == "nan":
            report.nan_count += 1
            invalid_row = True
        elif error == "infinity":
            report.infinity_count += 1
            invalid_row = True
        elif value is not None:
            values[numeric_field] = value
    if invalid_row:
        report.invalid_numeric_row_count += 1

    state.ignore_counts[row[11]] += 1
    if all(field in values for field in PRICE_FIELDS):
        if any(values[field] <= 0 for field in PRICE_FIELDS):
            report.nonpositive_price_row_count += 1
        if (
            values["high"] < values["open"]
            or values["high"] < values["close"]
            or values["high"] < values["low"]
            or values["low"] > values["open"]
            or values["low"] > values["close"]
            or values["low"] > values["high"]
        ):
            report.ohlc_violation_count += 1
    if all(field in values for field in VOLUME_FIELDS) and any(
        values[field] < 0 for field in VOLUME_FIELDS
    ):
        report.negative_volume_row_count += 1
    if values.get("volume") == 0:
        report.zero_volume_row_count += 1
    if "number_of_trades" in values and values["number_of_trades"] < 0:
        report.negative_trade_count_row_count += 1
    if (
        "taker_buy_base_volume" in values
        and "volume" in values
        and values["taker_buy_base_volume"] > values["volume"]
    ):
        report.taker_buy_base_exceeds_volume_count += 1
    if (
        "taker_buy_quote_volume" in values
        and "quote_asset_volume" in values
        and values["taker_buy_quote_volume"] > values["quote_asset_volume"]
    ):
        report.taker_buy_quote_exceeds_quote_volume_count += 1


def _record_timestamp_validation(
    row: Sequence[str],
    period: MonthlyPeriod,
    report: ArchiveValidation,
    state: _GlobalState,
    previous_archive_timestamp_us: int | None,
) -> tuple[int | None, int | None]:
    try:
        open_time = int(row[0])
        close_time = int(row[6])
    except ValueError:
        report.timestamp_parse_failure_count += 1
        return previous_archive_timestamp_us, None

    open_unit = _detect_unit(open_time)
    close_unit = _detect_unit(close_time)
    if open_unit is None or close_unit is None:
        report.timestamp_parse_failure_count += 1
        return previous_archive_timestamp_us, None
    if open_unit != report.expected_timestamp_unit or close_unit != report.expected_timestamp_unit:
        report.timestamp_unit_mismatch_count += 1

    open_time_us = _normalize_to_microseconds(open_time, open_unit)
    close_time_us = _normalize_to_microseconds(close_time, close_unit)
    try:
        open_datetime = _utc_datetime(open_time_us)
        _utc_datetime(close_time_us)
    except (OverflowError, ValueError):
        report.timestamp_parse_failure_count += 1
        return previous_archive_timestamp_us, None

    if report.first_open_time_utc is None:
        report.first_open_time_utc = _timestamp_text(open_time_us)
    report.last_open_time_utc = _timestamp_text(open_time_us)
    is_off_grid = open_time_us % INTERVAL_MICROSECONDS != 0
    if is_off_grid:
        report.alignment_violation_count += 1
        report.alignment_violation_open_times.append(_timestamp_text(open_time_us))

    source_interval = 300_000 if report.expected_timestamp_unit == "milliseconds" else 300_000_000
    expected_close_time = open_time + source_interval - 1
    is_close_time_violation = close_time != expected_close_time
    if is_close_time_violation:
        report.close_time_violation_count += 1
        report.close_time_violation_details.append(
            {
                "open_time_raw": open_time,
                "close_time_raw": close_time,
                "expected_close_time_raw": expected_close_time,
            }
        )
    if is_off_grid or is_close_time_violation:
        volume, _ = _decimal(row[5])
        state.timestamp_anomalies.append(
            _TimestampObservation(
                archive_month=period.key,
                raw_open_time=open_time,
                timestamp_unit=open_unit,
                normalized_open_time_us=open_time_us,
                is_off_grid=is_off_grid,
                actual_close_time_raw=close_time,
                expected_close_time_raw=expected_close_time,
                is_close_time_violation=is_close_time_violation,
                zero_volume=volume == 0,
                ignore_value=row[11],
            )
        )
    if previous_archive_timestamp_us is not None and open_time_us < previous_archive_timestamp_us:
        report.ordering_violation_count += 1
    if state.last_timestamp_us is not None and open_time_us < state.last_timestamp_us:
        state.global_ordering_violation_count += 1
    previous_archive_timestamp_us = open_time_us
    state.last_timestamp_us = open_time_us

    if (open_datetime.year, open_datetime.month) != (period.year, period.month):
        report.month_containment_violation_count += 1
        report.month_containment_violation_open_times.append(_timestamp_text(open_time_us))

    digest = _row_digest(row)
    previous = state.seen_rows.get(open_time_us)
    if previous is not None:
        state.duplicate_timestamp_count += 1
        previous_digest, previous_month = previous
        if digest == previous_digest:
            state.exact_duplicate_row_count += 1
        else:
            state.conflicting_duplicate_row_count += 1
        if previous_month != period.key:
            state.cross_month_duplicate_timestamp_count += 1
    else:
        state.seen_rows[open_time_us] = (digest, period.key)
    return previous_archive_timestamp_us, open_time_us


def _finalize_archive_status(report: ArchiveValidation) -> None:
    schema_invalid = (
        report.archive_open_status != "OPENED"
        or not report.expected_payload_exists
        or report.unexpected_zip_member_count > 0
        or report.unexpected_header_row_count > 0
        or report.column_count_problem_count > 0
    )
    parse_invalid = (
        report.archive_error is not None
        or report.malformed_row_count > 0
        or report.timestamp_parse_failure_count > 0
        or report.numeric_parse_failure_count > 0
        or report.nan_count > 0
        or report.infinity_count > 0
    )
    data_invalid = any(
        (
            report.timestamp_unit_mismatch_count,
            report.alignment_violation_count,
            report.close_time_violation_count,
            report.ordering_violation_count,
            report.month_containment_violation_count,
            report.nonpositive_price_row_count,
            report.ohlc_violation_count,
            report.negative_volume_row_count,
            report.negative_trade_count_row_count,
            report.taker_buy_base_exceeds_volume_count,
            report.taker_buy_quote_exceeds_quote_volume_count,
        )
    )
    report.schema_status = "INVALID" if schema_invalid else "VALID"
    report.parse_status = "INVALID" if parse_invalid else "VALID"
    report.validation_status = (
        "VALID"
        if report.schema_status == "VALID" and report.parse_status == "VALID" and not data_invalid
        else "INVALID"
    )


def _validate_archive(
    period: MonthlyPeriod,
    path: Path,
    expected_fields: Sequence[str],
    state: _GlobalState,
) -> _ArchiveOutcome:
    expected_member = f"{path.stem}.csv"
    report = ArchiveValidation(
        archive_month=period.key,
        archive_filename=path.name,
        expected_csv_member=expected_member,
        expected_timestamp_unit=_expected_unit(period),
    )
    minimum_timestamp_us: int | None = None
    maximum_timestamp_us: int | None = None
    previous_timestamp_us: int | None = None
    if not path.is_file():
        report.archive_open_status = "MISSING"
        report.archive_error = "Archive file is missing"
        _finalize_archive_status(report)
        return _ArchiveOutcome(report, None, None)

    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        report.archive_open_status = "ERROR"
        report.archive_error = str(exc)
        _finalize_archive_status(report)
        return _ArchiveOutcome(report, None, None)

    with archive:
        report.archive_open_status = "OPENED"
        member_names = [member.filename for member in archive.infolist()]
        report.zip_member_count = len(member_names)
        report.expected_payload_exists = expected_member in member_names
        unexpected_members = [name for name in member_names if name != expected_member]
        report.unexpected_zip_member_count = len(unexpected_members)
        report.unexpected_zip_members = json.dumps(unexpected_members, separators=(",", ":"))
        if not report.expected_payload_exists:
            report.archive_error = f"Expected CSV member {expected_member!r} is missing"
            _finalize_archive_status(report)
            return _ArchiveOutcome(report, None, None)

        try:
            with archive.open(expected_member, "r") as binary_handle:
                text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="")
                reader = csv.reader(text_handle, strict=True)
                for row in reader:
                    stripped = tuple(value.strip() for value in row)
                    if stripped == tuple(expected_fields):
                        report.unexpected_header_row_count += 1
                        continue
                    if len(row) != len(expected_fields):
                        report.malformed_row_count += 1
                        report.column_count_problem_count += 1
                        continue
                    report.row_count += 1
                    _record_numeric_validation(row, report, state)
                    previous_timestamp_us, timestamp_us = _record_timestamp_validation(
                        row, period, report, state, previous_timestamp_us
                    )
                    if timestamp_us is not None:
                        minimum_timestamp_us = (
                            timestamp_us
                            if minimum_timestamp_us is None
                            else min(minimum_timestamp_us, timestamp_us)
                        )
                        maximum_timestamp_us = (
                            timestamp_us
                            if maximum_timestamp_us is None
                            else max(maximum_timestamp_us, timestamp_us)
                        )
        except (OSError, UnicodeError, csv.Error, RuntimeError, zipfile.BadZipFile) as exc:
            report.malformed_row_count += 1
            report.archive_error = f"CSV payload could not be read completely: {exc}"

    _finalize_archive_status(report)
    return _ArchiveOutcome(report, minimum_timestamp_us, maximum_timestamp_us)


def _gap_rows(
    observed_timestamps_us: set[int],
    *,
    first_timestamp_us: int,
    cutoff_us: int,
) -> tuple[int, list[dict[str, object]]]:
    if first_timestamp_us >= cutoff_us:
        return 0, []
    expected_count = (
        cutoff_us - first_timestamp_us + INTERVAL_MICROSECONDS - 1
    ) // INTERVAL_MICROSECONDS
    gaps: list[dict[str, object]] = []
    gap_start: int | None = None
    gap_count = 0
    for index in range(expected_count):
        timestamp_us = first_timestamp_us + index * INTERVAL_MICROSECONDS
        if timestamp_us not in observed_timestamps_us:
            if gap_start is None:
                gap_start = timestamp_us
            gap_count += 1
        elif gap_start is not None:
            gaps.append(
                {
                    "gap_start": _timestamp_text(gap_start),
                    "gap_end": _timestamp_text(timestamp_us - INTERVAL_MICROSECONDS),
                    "missing_candles": gap_count,
                    "duration": f"PT{gap_count * 5}M",
                }
            )
            gap_start = None
            gap_count = 0
    if gap_start is not None:
        gaps.append(
            {
                "gap_start": _timestamp_text(gap_start),
                "gap_end": _timestamp_text(
                    first_timestamp_us + (expected_count - 1) * INTERVAL_MICROSECONDS
                ),
                "missing_candles": gap_count,
                "duration": f"PT{gap_count * 5}M",
            }
        )
    return expected_count, gaps


def _nearest_grid_details(timestamp_us: int) -> tuple[int, int | None, int]:
    lower = timestamp_us - timestamp_us % INTERVAL_MICROSECONDS
    upper = lower + INTERVAL_MICROSECONDS
    distance_to_lower = timestamp_us - lower
    distance_to_upper = upper - timestamp_us
    if distance_to_lower < distance_to_upper:
        return lower, None, distance_to_lower
    if distance_to_upper < distance_to_lower:
        return upper, None, -distance_to_upper
    return lower, upper, distance_to_lower


def _characterize_timestamp_anomalies(
    observations: Sequence[_TimestampObservation],
    *,
    observed_timestamps_us: set[int],
    first_timestamp_us: int | None,
    cutoff_us: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    target_by_index: dict[int, int] = {}
    alternate_by_index: dict[int, int] = {}
    signed_offset_by_index: dict[int, int] = {}
    target_counts: Counter[int] = Counter()
    for index, observation in enumerate(observations):
        if not observation.is_off_grid:
            continue
        target, alternate, signed_offset = _nearest_grid_details(
            observation.normalized_open_time_us
        )
        target_by_index[index] = target
        signed_offset_by_index[index] = signed_offset
        if alternate is not None:
            alternate_by_index[index] = alternate
        elif (
            first_timestamp_us is not None
            and first_timestamp_us <= target < cutoff_us
            and (target - first_timestamp_us) % INTERVAL_MICROSECONDS == 0
        ):
            target_counts[target] += 1

    rows: list[dict[str, object]] = []
    unique_missing_targets: set[int] = set()
    signed_offsets: list[int] = []
    absolute_offsets: list[int] = []
    affected_months: Counter[str] = Counter()
    mapping_counts: Counter[str] = Counter()
    close_error_counts: Counter[int] = Counter()
    overlap_count = 0
    close_only_count = 0
    for index, observation in enumerate(observations):
        scale = 1_000 if observation.timestamp_unit == "milliseconds" else 1
        target = target_by_index.get(index)
        alternate = alternate_by_index.get(index)
        signed_offset = signed_offset_by_index.get(index)
        target_absent: bool | None = None
        collides: bool | None = None
        mapping_status = "NOT_APPLICABLE"
        if observation.is_off_grid:
            affected_months[observation.archive_month] += 1
            assert target is not None
            assert signed_offset is not None
            signed_offsets.append(signed_offset)
            absolute_offsets.append(abs(signed_offset))
            if alternate is not None:
                mapping_status = "AMBIGUOUS_NEAREST_GRID_TIE"
            elif (
                first_timestamp_us is None
                or target < first_timestamp_us
                or target >= cutoff_us
                or (target - first_timestamp_us) % INTERVAL_MICROSECONDS != 0
            ):
                mapping_status = "OUTSIDE_EXPECTED_GRID"
            else:
                collides = target in observed_timestamps_us
                target_absent = not collides
                if collides:
                    mapping_status = "COLLIDES_WITH_EXISTING_GRID"
                elif target_counts[target] > 1:
                    mapping_status = "AMBIGUOUS_SHARED_TARGET"
                else:
                    mapping_status = "UNIQUE_MISSING_GRID"
                    unique_missing_targets.add(target)
            mapping_counts[mapping_status] += 1

        close_error_source_units = (
            observation.actual_close_time_raw - observation.expected_close_time_raw
        )
        close_error_us = close_error_source_units * scale
        if observation.is_close_time_violation:
            close_error_counts[close_error_us] += 1
            if observation.is_off_grid:
                overlap_count += 1
            else:
                close_only_count += 1
        anomaly_types = "+".join(
            name
            for name, present in (
                ("OFF_GRID", observation.is_off_grid),
                ("CLOSE_TIME", observation.is_close_time_violation),
            )
            if present
        )
        rows.append(
            {
                "archive_month": observation.archive_month,
                "raw_open_time": observation.raw_open_time,
                "timestamp_unit": observation.timestamp_unit,
                "normalized_open_time_utc": _timestamp_text(observation.normalized_open_time_us),
                "anomaly_types": anomaly_types,
                "nearest_grid_timestamp_utc": (
                    _timestamp_text(target) if target is not None else None
                ),
                "alternate_nearest_grid_timestamp_utc": (
                    _timestamp_text(alternate) if alternate is not None else None
                ),
                "signed_grid_offset_source_units": (
                    signed_offset // scale if signed_offset is not None else None
                ),
                "signed_grid_offset_microseconds": signed_offset,
                "absolute_grid_offset_microseconds": (
                    abs(signed_offset) if signed_offset is not None else None
                ),
                "nearest_grid_currently_absent": target_absent,
                "mapping_would_collide_with_existing": collides,
                "mapping_status": mapping_status,
                "actual_close_time_raw": observation.actual_close_time_raw,
                "expected_close_time_raw": observation.expected_close_time_raw,
                "signed_close_time_error_source_units": close_error_source_units,
                "signed_close_time_error_microseconds": close_error_us,
                "zero_volume": observation.zero_volume,
                "ignore_value": observation.ignore_value,
            }
        )

    hypothetical_gaps: list[dict[str, object]] = []
    if first_timestamp_us is not None:
        _, hypothetical_gaps = _gap_rows(
            observed_timestamps_us | unique_missing_targets,
            first_timestamp_us=first_timestamp_us,
            cutoff_us=cutoff_us,
        )
    hypothetical_longest_gap = max(
        hypothetical_gaps,
        key=lambda gap: int(gap["missing_candles"]),
        default=None,
    )
    summary = {
        "timestamp_anomaly_row_count": len(rows),
        "off_grid_affected_month_counts": dict(sorted(affected_months.items())),
        "off_grid_signed_offset_microseconds_counts": {
            str(offset): count for offset, count in sorted(Counter(signed_offsets).items())
        },
        "off_grid_signed_offset_microseconds_min": min(signed_offsets, default=0),
        "off_grid_signed_offset_microseconds_median": median(signed_offsets)
        if signed_offsets
        else 0,
        "off_grid_signed_offset_microseconds_max": max(signed_offsets, default=0),
        "off_grid_absolute_offset_microseconds_min": min(absolute_offsets, default=0),
        "off_grid_absolute_offset_microseconds_median": (
            median(absolute_offsets) if absolute_offsets else 0
        ),
        "off_grid_absolute_offset_microseconds_max": max(absolute_offsets, default=0),
        "off_grid_unique_missing_mapping_count": mapping_counts["UNIQUE_MISSING_GRID"],
        "off_grid_existing_grid_collision_count": mapping_counts["COLLIDES_WITH_EXISTING_GRID"],
        "off_grid_ambiguous_or_unmappable_count": sum(
            count
            for status, count in mapping_counts.items()
            if status not in {"UNIQUE_MISSING_GRID", "COLLIDES_WITH_EXISTING_GRID"}
        ),
        "close_time_overlap_off_grid_count": overlap_count,
        "close_time_only_violation_count": close_only_count,
        "close_time_error_microseconds_counts": {
            str(error): count for error, count in sorted(close_error_counts.items())
        },
        "candidate_true_missing_count": sum(
            int(gap["missing_candles"]) for gap in hypothetical_gaps
        ),
        "hypothetical_gap_episode_count": len(hypothetical_gaps),
        "hypothetical_longest_gap": hypothetical_longest_gap,
        "timestamp_anomaly_zero_volume_count": sum(
            observation.zero_volume for observation in observations
        ),
        "timestamp_anomaly_nonzero_ignore_count": sum(
            observation.ignore_value != "0" for observation in observations
        ),
    }
    return rows, summary, hypothetical_gaps


def validate_raw_archives(
    archives: Sequence[tuple[MonthlyPeriod, Path]],
    *,
    expected_fields: Sequence[str],
    cutoff: datetime,
) -> RawDatasetValidation:
    """Validate monthly Binance kline ZIPs without modifying or extracting them."""
    if cutoff.tzinfo is None or cutoff.utcoffset() != timedelta(0):
        raise ValueError("cutoff must be timezone-aware UTC")
    required_fields = (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    )
    if tuple(expected_fields) != required_fields:
        raise ValueError("expected_fields must match the exact 12-field Binance kline schema")

    state = _GlobalState(seen_rows={}, ignore_counts=Counter())
    outcomes = [
        _validate_archive(period, path, expected_fields, state) for period, path in archives
    ]
    reports = [outcome.report for outcome in outcomes]
    observed = set(state.seen_rows)
    first_timestamp_us = min(observed) if observed else None
    last_timestamp_us = max(observed) if observed else None
    cutoff_us = int((cutoff - EPOCH).total_seconds() * 1_000_000)
    expected_candle_count = 0
    gaps: list[dict[str, object]] = []
    if first_timestamp_us is not None:
        expected_candle_count, gaps = _gap_rows(
            observed, first_timestamp_us=first_timestamp_us, cutoff_us=cutoff_us
        )
    observed_unique_candle_count = 0
    if first_timestamp_us is not None:
        observed_unique_candle_count = sum(
            first_timestamp_us <= timestamp_us < cutoff_us
            and (timestamp_us - first_timestamp_us) % INTERVAL_MICROSECONDS == 0
            for timestamp_us in observed
        )
    missing_candle_count = sum(int(gap["missing_candles"]) for gap in gaps)
    timestamp_anomaly_rows, timestamp_characterization, _ = _characterize_timestamp_anomalies(
        state.timestamp_anomalies,
        observed_timestamps_us=observed,
        first_timestamp_us=first_timestamp_us,
        cutoff_us=cutoff_us,
    )

    cross_month_overlap_count = 0
    cross_month_gap_count = 0
    cross_month_gap_missing_candle_count = 0
    for previous, current in pairwise(outcomes):
        if previous.maximum_timestamp_us is None or current.minimum_timestamp_us is None:
            continue
        if current.minimum_timestamp_us <= previous.maximum_timestamp_us:
            cross_month_overlap_count += 1
        elif current.minimum_timestamp_us > previous.maximum_timestamp_us + INTERVAL_MICROSECONDS:
            cross_month_gap_count += 1
            cross_month_gap_missing_candle_count += (
                current.minimum_timestamp_us - previous.maximum_timestamp_us
            ) // INTERVAL_MICROSECONDS - 1

    def total(field: str) -> int:
        return sum(int(getattr(report, field)) for report in reports)

    valid_archive_count = sum(report.validation_status == "VALID" for report in reports)
    invalid_archive_count = len(reports) - valid_archive_count
    longest_gap = max(gaps, key=lambda gap: int(gap["missing_candles"]), default=None)
    execution_status = "COMPLETE" if first_timestamp_us is not None else "BLOCKED"
    review_required = (
        invalid_archive_count > 0
        or state.duplicate_timestamp_count > 0
        or missing_candle_count > 0
        or state.global_ordering_violation_count > 0
    )
    quality_decision = (
        "BLOCKED"
        if execution_status == "BLOCKED"
        else "REVIEW_REQUIRED"
        if review_required
        else "CLEAN"
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "phase": "E00B",
        "execution_status": execution_status,
        "dataset_quality_decision": quality_decision,
        "archives_scanned": len(reports),
        "archives_opened": sum(report.archive_open_status == "OPENED" for report in reports),
        "valid_archive_count": valid_archive_count,
        "invalid_archive_count": invalid_archive_count,
        "schema_valid_archive_count": sum(report.schema_status == "VALID" for report in reports),
        "schema_invalid_archive_count": sum(
            report.schema_status == "INVALID" for report in reports
        ),
        "parse_valid_archive_count": sum(report.parse_status == "VALID" for report in reports),
        "parse_invalid_archive_count": sum(report.parse_status == "INVALID" for report in reports),
        "archive_open_failure_count": sum(
            report.archive_open_status != "OPENED" for report in reports
        ),
        "missing_payload_count": sum(not report.expected_payload_exists for report in reports),
        "unexpected_zip_member_count": total("unexpected_zip_member_count"),
        "malformed_row_count": total("malformed_row_count"),
        "column_count_problem_count": total("column_count_problem_count"),
        "unexpected_header_row_count": total("unexpected_header_row_count"),
        "total_row_count": total("row_count"),
        "actual_first_candle": (
            _timestamp_text(first_timestamp_us) if first_timestamp_us is not None else None
        ),
        "actual_last_candle": (
            _timestamp_text(last_timestamp_us) if last_timestamp_us is not None else None
        ),
        "expected_cutoff_exclusive": cutoff.isoformat().replace("+00:00", "Z"),
        "millisecond_archive_count": sum(
            report.expected_timestamp_unit == "milliseconds" for report in reports
        ),
        "microsecond_archive_count": sum(
            report.expected_timestamp_unit == "microseconds" for report in reports
        ),
        "timestamp_parse_failure_count": total("timestamp_parse_failure_count"),
        "timestamp_unit_mismatch_count": total("timestamp_unit_mismatch_count"),
        "timestamp_unit_mismatch_archive_count": sum(
            report.timestamp_unit_mismatch_count > 0 for report in reports
        ),
        "alignment_violation_count": total("alignment_violation_count"),
        "close_time_violation_count": total("close_time_violation_count"),
        "archive_ordering_violation_count": total("ordering_violation_count"),
        "global_ordering_violation_count": state.global_ordering_violation_count,
        "global_ordering_monotonic": state.global_ordering_violation_count == 0,
        "duplicate_timestamp_count": state.duplicate_timestamp_count,
        "exact_duplicate_row_count": state.exact_duplicate_row_count,
        "conflicting_duplicate_row_count": state.conflicting_duplicate_row_count,
        "expected_candle_count": expected_candle_count,
        "observed_unique_candle_count": observed_unique_candle_count,
        "observed_unique_timestamp_count": len(observed),
        "missing_candle_count": missing_candle_count,
        "missing_percentage": (
            missing_candle_count / expected_candle_count * 100 if expected_candle_count else 0.0
        ),
        "gap_episode_count": len(gaps),
        "longest_gap_candles": int(longest_gap["missing_candles"]) if longest_gap else 0,
        "longest_gap_duration": longest_gap["duration"] if longest_gap else "PT0M",
        "longest_gap": longest_gap,
        "first_gap": gaps[0] if gaps else None,
        "last_gap": gaps[-1] if gaps else None,
        "numeric_parse_failure_count": total("numeric_parse_failure_count"),
        "nan_count": total("nan_count"),
        "infinity_count": total("infinity_count"),
        "invalid_numeric_row_count": total("invalid_numeric_row_count"),
        "nonpositive_price_row_count": total("nonpositive_price_row_count"),
        "ohlc_violation_count": total("ohlc_violation_count"),
        "negative_volume_row_count": total("negative_volume_row_count"),
        "zero_volume_row_count": total("zero_volume_row_count"),
        "negative_trade_count_row_count": total("negative_trade_count_row_count"),
        "taker_buy_base_exceeds_volume_count": total("taker_buy_base_exceeds_volume_count"),
        "taker_buy_quote_exceeds_quote_volume_count": total(
            "taker_buy_quote_exceeds_quote_volume_count"
        ),
        "month_containment_violation_count": total("month_containment_violation_count"),
        "cross_month_overlap_count": cross_month_overlap_count,
        "cross_month_gap_count": cross_month_gap_count,
        "cross_month_gap_missing_candle_count": cross_month_gap_missing_candle_count,
        "cross_month_duplicate_timestamp_count": state.cross_month_duplicate_timestamp_count,
        "ignore_unique_value_count": len(state.ignore_counts),
        "ignore_zero_count": state.ignore_counts.get("0", 0),
        "ignore_nonzero_count": sum(
            count for value, count in state.ignore_counts.items() if value != "0"
        ),
        "ignore_value_counts": dict(sorted(state.ignore_counts.items())),
        "numeric_comparison_tolerance": "none; decimal source values compared exactly",
        "duplicate_comparison": "SHA-256 fingerprint of the exact 12 parsed CSV fields",
        "raw_data_modified": False,
    }
    summary.update(timestamp_characterization)
    return RawDatasetValidation(
        summary=summary,
        archive_rows=[report.to_dict() for report in reports],
        gaps=gaps,
        timestamp_anomalies=timestamp_anomaly_rows,
    )


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def run_raw_validation(
    *,
    project_root: Path,
    config_path: Path,
    config: AcquisitionConfig,
) -> RawValidationRunResult:
    """Run E00B against the local immutable archive set and write compact artifacts."""
    root = project_root.resolve()
    periods = generate_monthly_periods(config.start_date, config.end_date_exclusive)
    archives = [
        (
            period,
            root / config.destination / archive_filename(config.symbol, config.interval, period),
        )
        for period in periods
    ]
    cutoff = datetime.combine(config.end_date_exclusive, datetime.min.time(), tzinfo=UTC)
    validation = validate_raw_archives(
        archives, expected_fields=config.expected_kline_fields, cutoff=cutoff
    )
    output_directory = root / "outputs" / "data" / "e00b"
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "raw_validation_summary.json"
    archive_validation_path = output_directory / "archive_validation.csv"
    gaps_path = output_directory / "gaps.csv"
    timestamp_anomalies_path = output_directory / "timestamp_anomalies.csv"

    summary = dict(validation.summary)
    summary["generated_at_utc"] = datetime.now(UTC).isoformat()
    summary["config_path"] = config_path.resolve().relative_to(root).as_posix()
    _write_csv(
        archive_validation_path,
        validation.archive_rows,
        list(ArchiveValidation.__dataclass_fields__),
    )
    _write_csv(
        gaps_path,
        validation.gaps,
        ("gap_start", "gap_end", "missing_candles", "duration"),
    )
    _write_csv(
        timestamp_anomalies_path,
        validation.timestamp_anomalies,
        TIMESTAMP_ANOMALY_FIELDS,
    )
    _write_json(summary_path, summary)
    return RawValidationRunResult(
        exit_code=0 if summary["execution_status"] == "COMPLETE" else 1,
        summary=summary,
        summary_path=summary_path,
        archive_validation_path=archive_validation_path,
        gaps_path=gaps_path,
        timestamp_anomalies_path=timestamp_anomalies_path,
    )
