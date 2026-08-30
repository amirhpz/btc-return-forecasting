from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from btc_forecasting.data.binance_archive import MonthlyPeriod, archive_filename
from btc_forecasting.data.raw_validation import validate_raw_archives

KLINE_FIELDS = (
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
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _timestamp(value: datetime, unit: str) -> int:
    microseconds = int((value - EPOCH).total_seconds() * 1_000_000)
    return microseconds // 1_000 if unit == "milliseconds" else microseconds


def _row(
    value: datetime,
    *,
    unit: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    open_time = _timestamp(value, unit)
    interval = 300_000 if unit == "milliseconds" else 300_000_000
    values = {
        "open_time": str(open_time),
        "open": "100",
        "high": "110",
        "low": "90",
        "close": "105",
        "volume": "10",
        "close_time": str(open_time + interval - 1),
        "quote_asset_volume": "1000",
        "number_of_trades": "5",
        "taker_buy_base_volume": "4",
        "taker_buy_quote_volume": "400",
        "ignore": "0",
    }
    values.update(overrides or {})
    return [values[field] for field in KLINE_FIELDS]


def _archive(
    root: Path,
    period: MonthlyPeriod,
    rows: list[list[str] | tuple[str, ...]],
    *,
    extra_member: bool = False,
) -> Path:
    filename = archive_filename("BTCUSDT", "5m", period)
    path = root / filename
    member_name = f"{path.stem}.csv"
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, text.getvalue())
        if extra_member:
            archive.writestr("unexpected.txt", "unexpected")
    return path


def _validate(
    archives: list[tuple[MonthlyPeriod, Path]],
    *,
    cutoff: datetime,
):
    return validate_raw_archives(
        archives,
        expected_fields=KLINE_FIELDS,
        cutoff=cutoff,
    )


def test_valid_12_column_millisecond_archive_before_2025(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 12)
    start = datetime(2024, 12, 1, tzinfo=UTC)
    path = _archive(
        tmp_path,
        period,
        [_row(start, unit="milliseconds"), _row(start + timedelta(minutes=5), unit="milliseconds")],
    )

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=10))

    assert result.summary["valid_archive_count"] == 1
    assert result.summary["schema_valid_archive_count"] == 1
    assert result.summary["millisecond_archive_count"] == 1
    assert result.summary["total_row_count"] == 2
    assert result.summary["missing_candle_count"] == 0
    assert result.summary["ignore_value_counts"] == {"0": 2}


def test_microsecond_archive_from_2025(tmp_path: Path) -> None:
    period = MonthlyPeriod(2025, 1)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    path = _archive(tmp_path, period, [_row(start, unit="microseconds")])

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=5))

    assert result.summary["valid_archive_count"] == 1
    assert result.summary["microsecond_archive_count"] == 1
    assert result.summary["timestamp_unit_mismatch_count"] == 0


def test_unit_mismatch_alignment_and_close_time_are_reported(tmp_path: Path) -> None:
    period = MonthlyPeriod(2025, 1)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    mismatched = _row(start, unit="milliseconds")
    misaligned = _row(start + timedelta(minutes=1), unit="microseconds")
    path = _archive(tmp_path, period, [mismatched, misaligned])

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=10))

    assert result.summary["timestamp_unit_mismatch_count"] == 1
    assert result.summary["alignment_violation_count"] == 1
    assert result.summary["close_time_violation_count"] == 1
    archive = result.archive_rows[0]
    assert json.loads(archive["alignment_violation_open_times"]) == ["2025-01-01T00:01:00Z"]
    assert len(json.loads(archive["close_time_violation_details"])) == 1


def test_timestamp_anomaly_mapping_and_close_time_overlap(tmp_path: Path) -> None:
    period = MonthlyPeriod(2025, 1)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    collision = _row(start + timedelta(minutes=1), unit="microseconds")
    unique_missing = _row(start + timedelta(minutes=6), unit="microseconds")
    unique_missing[6] = str(int(unique_missing[6]) + 7)
    ambiguous = _row(start + timedelta(minutes=12, seconds=30), unit="microseconds")
    close_only = _row(start + timedelta(minutes=20), unit="microseconds")
    close_only[6] = str(int(close_only[6]) + 11)
    path = _archive(
        tmp_path,
        period,
        [
            _row(start, unit="microseconds"),
            collision,
            unique_missing,
            ambiguous,
            close_only,
        ],
    )

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=25))
    summary = result.summary

    assert summary["missing_candle_count"] == 3
    assert summary["off_grid_unique_missing_mapping_count"] == 1
    assert summary["off_grid_existing_grid_collision_count"] == 1
    assert summary["off_grid_ambiguous_or_unmappable_count"] == 1
    assert summary["close_time_overlap_off_grid_count"] == 1
    assert summary["close_time_only_violation_count"] == 1
    assert summary["candidate_true_missing_count"] == 2
    assert summary["hypothetical_gap_episode_count"] == 1
    assert summary["hypothetical_longest_gap"]["missing_candles"] == 2
    assert summary["off_grid_signed_offset_microseconds_counts"] == {
        "60000000": 2,
        "150000000": 1,
    }
    assert summary["close_time_error_microseconds_counts"] == {"7": 1, "11": 1}
    assert {row["mapping_status"] for row in result.timestamp_anomalies} == {
        "AMBIGUOUS_NEAREST_GRID_TIE",
        "COLLIDES_WITH_EXISTING_GRID",
        "NOT_APPLICABLE",
        "UNIQUE_MISSING_GRID",
    }


def test_duplicate_timestamps_distinguish_exact_and_conflicting_rows(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 1)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    original = _row(start, unit="milliseconds")
    conflicting = _row(start, unit="milliseconds", overrides={"close": "106"})
    path = _archive(tmp_path, period, [original, original, conflicting])

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=5))

    assert result.summary["duplicate_timestamp_count"] == 2
    assert result.summary["exact_duplicate_row_count"] == 1
    assert result.summary["conflicting_duplicate_row_count"] == 1


def test_contiguous_missing_candles_form_one_gap_episode(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 1)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    path = _archive(
        tmp_path,
        period,
        [
            _row(start, unit="milliseconds"),
            _row(start + timedelta(minutes=15), unit="milliseconds"),
        ],
    )

    result = _validate([(period, path)], cutoff=start + timedelta(minutes=20))

    assert result.summary["expected_candle_count"] == 4
    assert result.summary["missing_candle_count"] == 2
    assert result.summary["gap_episode_count"] == 1
    assert result.gaps == [
        {
            "gap_start": "2024-01-01T00:05:00Z",
            "gap_end": "2024-01-01T00:10:00Z",
            "missing_candles": 2,
            "duration": "PT10M",
        }
    ]


def test_cross_month_gap_and_duplicate_are_reported(tmp_path: Path) -> None:
    gap_root = tmp_path / "gap"
    gap_root.mkdir()
    january = MonthlyPeriod(2024, 1)
    february = MonthlyPeriod(2024, 2)
    january_end = datetime(2024, 1, 31, 23, 50, tzinfo=UTC)
    february_start = datetime(2024, 2, 1, 0, 5, tzinfo=UTC)
    january_path = _archive(gap_root, january, [_row(january_end, unit="milliseconds")])
    february_path = _archive(gap_root, february, [_row(february_start, unit="milliseconds")])

    gap_result = _validate(
        [(january, january_path), (february, february_path)],
        cutoff=february_start + timedelta(minutes=5),
    )
    assert gap_result.summary["cross_month_gap_count"] == 1
    assert gap_result.summary["cross_month_gap_missing_candle_count"] == 2

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    boundary = datetime(2024, 1, 31, 23, 55, tzinfo=UTC)
    boundary_row = _row(boundary, unit="milliseconds")
    january_path = _archive(duplicate_root, january, [boundary_row])
    february_path = _archive(duplicate_root, february, [boundary_row])
    duplicate_result = _validate(
        [(january, january_path), (february, february_path)],
        cutoff=datetime(2024, 2, 1, 0, 5, tzinfo=UTC),
    )
    assert duplicate_result.summary["cross_month_overlap_count"] == 1
    assert duplicate_result.summary["cross_month_duplicate_timestamp_count"] == 1
    assert duplicate_result.summary["month_containment_violation_count"] == 1


def test_numeric_ohlc_and_taker_volume_violations_are_reported(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 1)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    invalid_numeric = _row(
        start,
        unit="milliseconds",
        overrides={"open": "NaN", "high": "Infinity", "low": "not-a-number"},
    )
    invalid_ohlc = _row(
        start + timedelta(minutes=5),
        unit="milliseconds",
        overrides={"open": "0", "high": "90", "low": "100"},
    )
    invalid_volume = _row(
        start + timedelta(minutes=10),
        unit="milliseconds",
        overrides={
            "volume": "-1",
            "quote_asset_volume": "-2",
            "number_of_trades": "-1",
            "taker_buy_base_volume": "2",
            "taker_buy_quote_volume": "3",
        },
    )
    path = _archive(tmp_path, period, [invalid_numeric, invalid_ohlc, invalid_volume])

    summary = _validate([(period, path)], cutoff=start + timedelta(minutes=15)).summary

    assert summary["numeric_parse_failure_count"] == 1
    assert summary["nan_count"] == 1
    assert summary["infinity_count"] == 1
    assert summary["invalid_numeric_row_count"] == 1
    assert summary["nonpositive_price_row_count"] == 1
    assert summary["ohlc_violation_count"] == 1
    assert summary["negative_volume_row_count"] == 1
    assert summary["negative_trade_count_row_count"] == 1
    assert summary["taker_buy_base_exceeds_volume_count"] == 1
    assert summary["taker_buy_quote_exceeds_quote_volume_count"] == 1


def test_headers_column_problems_extra_members_and_month_leakage_are_reported(
    tmp_path: Path,
) -> None:
    period = MonthlyPeriod(2024, 1)
    leaked = datetime(2024, 2, 1, tzinfo=UTC)
    path = _archive(
        tmp_path,
        period,
        [KLINE_FIELDS, ["too", "short"], _row(leaked, unit="milliseconds")],
        extra_member=True,
    )

    result = _validate([(period, path)], cutoff=leaked + timedelta(minutes=5))
    archive = result.archive_rows[0]

    assert archive["unexpected_header_row_count"] == 1
    assert archive["malformed_row_count"] == 1
    assert archive["column_count_problem_count"] == 1
    assert archive["unexpected_zip_member_count"] == 1
    assert archive["month_containment_violation_count"] == 1
    assert archive["schema_status"] == "INVALID"
    assert result.summary["schema_invalid_archive_count"] == 1
