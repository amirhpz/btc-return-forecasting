from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow as pa

from btc_forecasting.data.canonical_1h import resample_complete_hours
from btc_forecasting.data.canonical_5m import CANONICAL_SCHEMA


def _source_row(open_time: datetime, index: int) -> dict[str, object]:
    value = Decimal(index + 1)
    return {
        "open_time": open_time,
        "open": Decimal("100.00000000") + value,
        "high": Decimal("101.00000000") + value,
        "low": Decimal("99.00000000") - value,
        "close": Decimal("100.50000000") + value,
        "volume": value,
        "close_time": open_time + timedelta(minutes=5) - timedelta(microseconds=1),
        "quote_asset_volume": value * Decimal(10),
        "number_of_trades": index + 1,
        "taker_buy_base_volume": Decimal("0.50000000"),
        "taker_buy_quote_volume": Decimal("5.00000000"),
        "ignore": str(index),
    }


def _source_table(timestamps: list[datetime]) -> pa.Table:
    return pa.Table.from_pylist(
        [_source_row(timestamp, index) for index, timestamp in enumerate(timestamps)],
        schema=CANONICAL_SCHEMA,
    )


def test_hourly_ohlc_and_additive_aggregation_is_exact() -> None:
    hour = datetime(2026, 1, 1, tzinfo=UTC)
    source = _source_table([hour + timedelta(minutes=5 * index) for index in range(12)])

    row = resample_complete_hours(source).table.to_pylist()[0]

    assert row == {
        "open_time": hour,
        "open": Decimal("101.00000000"),
        "high": Decimal("113.00000000"),
        "low": Decimal("87.00000000"),
        "close": Decimal("112.50000000"),
        "volume": Decimal("78.00000000"),
        "quote_asset_volume": Decimal("780.00000000"),
        "number_of_trades": 78,
        "taker_buy_base_volume": Decimal("6.00000000"),
        "taker_buy_quote_volume": Decimal("60.00000000"),
    }
    assert "close_time" not in row
    assert "ignore" not in row


def test_complete_twelve_candle_hour_is_retained() -> None:
    hour = datetime(2026, 1, 1, 1, tzinfo=UTC)
    source = _source_table([hour + timedelta(minutes=5 * index) for index in range(12)])

    result = resample_complete_hours(source)

    assert result.total_possible_hours == 1
    assert result.table.num_rows == 1
    assert result.incomplete_hours == []
    assert result.every_retained_hour_complete is True


def test_incomplete_hour_is_excluded_and_recorded() -> None:
    first_hour = datetime(2026, 1, 1, tzinfo=UTC)
    second_hour = first_hour + timedelta(hours=1)
    timestamps = [
        first_hour + timedelta(minutes=5 * index) for index in range(12) if index != 5
    ]
    timestamps.extend(
        second_hour + timedelta(minutes=5 * index) for index in range(12)
    )

    result = resample_complete_hours(_source_table(timestamps))

    assert result.total_possible_hours == 2
    assert result.table.column("open_time").to_pylist() == [second_hour]
    assert result.incomplete_hours == [
        {
            "open_time": "2026-01-01T00:00:00Z",
            "source_candle_count": 11,
            "missing_timestamp_count": 1,
            "missing_timestamps": "2026-01-01T00:25:00Z",
        }
    ]
