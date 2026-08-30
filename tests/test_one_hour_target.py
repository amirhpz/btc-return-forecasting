from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow as pa

from btc_forecasting.data.canonical_1h import CANONICAL_1H_SCHEMA
from btc_forecasting.targets.one_hour import construct_one_hour_targets


def _row(open_time: datetime, close: str) -> dict[str, object]:
    close_value = Decimal(close)
    return {
        "open_time": open_time,
        "open": close_value,
        "high": close_value,
        "low": close_value,
        "close": close_value,
        "volume": Decimal("1.00000000"),
        "quote_asset_volume": Decimal("100.00000000"),
        "number_of_trades": 1,
        "taker_buy_base_volume": Decimal("0.50000000"),
        "taker_buy_quote_volume": Decimal("50.00000000"),
    }


def _table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CANONICAL_1H_SCHEMA)


def test_one_hour_future_log_return_and_temporal_semantics() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    source = _table(
        [
            _row(start, "100.00000000"),
            _row(start + timedelta(hours=1), "110.00000000"),
        ]
    )

    result = construct_one_hour_targets(source)
    target = result.table.to_pylist()[0]

    assert math.isclose(target["future_log_return_1h"], math.log(1.1))
    assert target["bar_open_time"] == start
    assert target["decision_time"] == start + timedelta(hours=1)
    assert target["target_time"] == start + timedelta(hours=2)
    assert result.input_row_count == 2
    assert result.missing_next_hour_exclusion_count == 0


def test_no_target_is_created_across_an_hourly_gap() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    source = _table(
        [
            _row(start, "100.00000000"),
            _row(start + timedelta(hours=2), "110.00000000"),
        ]
    )

    result = construct_one_hour_targets(source)

    assert result.table.num_rows == 0
    assert result.missing_next_hour_exclusion_count == 1
    assert result.final_row_exclusion_count == 1


def test_final_row_has_no_fabricated_target() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    result = construct_one_hour_targets(_table([_row(start, "100.00000000")]))

    assert result.table.num_rows == 0
    assert result.missing_next_hour_exclusion_count == 0
    assert result.final_row_exclusion_count == 1
