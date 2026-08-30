from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from btc_forecasting.data.binance_archive import MonthlyPeriod
from btc_forecasting.data.canonical_5m import (
    TimestampCorrection,
    build_canonical_dataset,
    load_timestamp_corrections,
)

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


def _milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _row(
    open_time: datetime,
    *,
    close_time: datetime | None = None,
    volume: str = "1.00000000",
    ignore: str = "0",
) -> list[str]:
    open_time_raw = _milliseconds(open_time)
    close_time_raw = (
        _milliseconds(close_time)
        if close_time is not None
        else open_time_raw + 300_000 - 1
    )
    return [
        str(open_time_raw),
        "100.00000000",
        "102.00000000",
        "99.00000000",
        "101.00000000",
        volume,
        str(close_time_raw),
        "50.00000000",
        "7",
        "0.40000000",
        "40.00000000",
        ignore,
    ]


def _archive(tmp_path: Path, period: MonthlyPeriod, rows: list[list[str]]) -> Path:
    path = tmp_path / f"BTCUSDT-5m-{period.key}.zip"
    payload = io.StringIO(newline="")
    csv.writer(payload, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.stem}.csv", payload.getvalue())
    return path


def test_build_corrects_only_allow_list_and_preserves_source_payload(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 1)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    off_grid = start + timedelta(minutes=6)
    source_close_time = start + timedelta(minutes=8)
    archive = _archive(
        tmp_path,
        period,
        [
            _row(start),
            _row(
                off_grid,
                close_time=source_close_time,
                volume="0.00000000",
                ignore="17",
            ),
            _row(start + timedelta(minutes=15)),
        ],
    )
    correction = TimestampCorrection(
        archive_month=period.key,
        raw_open_time=_milliseconds(off_grid),
        corrected_open_time_us=int((start + timedelta(minutes=5)).timestamp() * 1_000_000),
    )
    output_path = tmp_path / "canonical.parquet"

    result = build_canonical_dataset(
        [(period, archive)],
        expected_fields=KLINE_FIELDS,
        corrections=[correction],
        output_path=output_path,
        cutoff=start + timedelta(minutes=20),
        expected_correction_count=1,
        expected_missing_grid_positions=1,
    )

    assert result.verification == {
        "row_count": 3,
        "first_open_time": "2024-01-01T00:00:00Z",
        "last_open_time": "2024-01-01T00:15:00Z",
        "grid_alignment_violation_count": 0,
        "timestamps_strictly_ordered": True,
        "duplicate_open_time_count": 0,
        "missing_grid_position_count": 1,
        "timestamp_correction_count": 1,
        "raw_archive_modified_count": 0,
    }
    table = pq.read_table(output_path)
    assert table.column("open_time").to_pylist() == [
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=15),
    ]
    assert table.column("close_time").to_pylist()[1] == source_close_time
    assert table.column("volume").to_pylist()[1] == Decimal("0.00000000")
    assert table.column("ignore").to_pylist()[1] == "17"


def test_build_rejects_an_off_grid_timestamp_not_in_allow_list(tmp_path: Path) -> None:
    period = MonthlyPeriod(2024, 1)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    archive = _archive(tmp_path, period, [_row(start + timedelta(minutes=1))])

    with pytest.raises(ValueError, match="Unapproved off-grid open_time"):
        build_canonical_dataset(
            [(period, archive)],
            expected_fields=KLINE_FIELDS,
            corrections=[],
            output_path=tmp_path / "canonical.parquet",
            cutoff=start + timedelta(minutes=5),
            expected_correction_count=0,
            expected_missing_grid_positions=0,
        )


def test_load_corrections_uses_only_collision_free_off_grid_rows(tmp_path: Path) -> None:
    path = tmp_path / "timestamp_anomalies.csv"
    fields = (
        "archive_month",
        "raw_open_time",
        "anomaly_types",
        "nearest_grid_timestamp_utc",
        "nearest_grid_currently_absent",
        "mapping_would_collide_with_existing",
        "mapping_status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "archive_month": "2018-02",
                "raw_open_time": "1518170294789",
                "anomaly_types": "OFF_GRID+CLOSE_TIME",
                "nearest_grid_timestamp_utc": "2018-02-09T10:00:00Z",
                "nearest_grid_currently_absent": "True",
                "mapping_would_collide_with_existing": "False",
                "mapping_status": "UNIQUE_MISSING_GRID",
            }
        )
        writer.writerow(
            {
                "archive_month": "2017-09",
                "raw_open_time": "1504713300000",
                "anomaly_types": "CLOSE_TIME",
                "nearest_grid_timestamp_utc": "",
                "nearest_grid_currently_absent": "",
                "mapping_would_collide_with_existing": "",
                "mapping_status": "NOT_APPLICABLE",
            }
        )

    corrections = load_timestamp_corrections(path, expected_count=1)

    assert corrections == (
        TimestampCorrection(
            archive_month="2018-02",
            raw_open_time=1518170294789,
            corrected_open_time_us=1518170400000000,
        ),
    )
