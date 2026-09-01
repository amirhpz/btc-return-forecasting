from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from btc_forecasting.data.canonical_1h import CANONICAL_1H_SCHEMA
from btc_forecasting.targets.horizon_audit import (
    FROZEN_HORIZONS_HOURS,
    _read_existing_train_targets,
    _sample_identity,
    common_anchor_times,
    construct_horizon_targets,
    normalized_diagnostic,
    one_hour_regression_check,
)
from btc_forecasting.targets.one_hour import construct_one_hour_targets


def _canonical(times: list[pd.Timestamp], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"open_time": times, "close": closes})


def _arrow_row(open_time: datetime, close: str) -> dict[str, object]:
    value = Decimal(close)
    return {
        "open_time": open_time,
        "open": value,
        "high": value,
        "low": value,
        "close": value,
        "volume": Decimal("1.00000000"),
        "quote_asset_volume": Decimal("100.00000000"),
        "number_of_trades": 1,
        "taker_buy_base_volume": Decimal("0.50000000"),
        "taker_buy_quote_volume": Decimal("50.00000000"),
    }


def test_frozen_horizons_are_exact() -> None:
    assert FROZEN_HORIZONS_HOURS == (1, 3, 6, 12)


def test_exact_endpoint_alignment_and_return_formula() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    times = [start + offset * timedelta(hours=1) for offset in range(5)]
    source = _canonical(times, [100.0, 101.0, 102.0, 110.0, 111.0])

    result = construct_horizon_targets(
        source,
        pd.DatetimeIndex([start]),
        horizon_hours=3,
        target_scope_end_exclusive=(start + timedelta(hours=10)).to_pydatetime(),
    )

    row = result.valid.iloc[0]
    assert row["feature_bar_time"] == start
    assert row["decision_time"] == start + timedelta(hours=1)
    assert row["actual_endpoint_time"] == start + timedelta(hours=3)
    assert row["target_time"] == start + timedelta(hours=4)
    assert row["future_log_return"] == pytest.approx(math.log(1.1))
    assert row["complete_consecutive_path"]


def test_one_hour_logic_reproduces_existing_target_semantics() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _arrow_row(start + timedelta(hours=offset), str(100 + offset))
        for offset in range(12)
    ]
    existing = construct_one_hour_targets(
        pa.Table.from_pylist(rows, schema=CANONICAL_1H_SCHEMA)
    ).table.to_pandas()
    canonical = pd.DataFrame(
        {
            "open_time": [row["open_time"] for row in rows],
            "close": [float(row["close"]) for row in rows],
        }
    )
    anchors = pd.DatetimeIndex(pd.to_datetime(canonical["open_time"], utc=True))[:-1]
    construction = construct_horizon_targets(
        canonical,
        anchors,
        horizon_hours=1,
        target_scope_end_exclusive=start + timedelta(hours=13),
    )

    check = one_hour_regression_check(construction, existing)

    assert check["common_row_count"] == 11
    assert check["timestamp_mismatch_count"] == 0
    assert check["numerically_equal_within_tolerance"] is True


def test_missing_intermediate_hour_invalidates_target_without_fill() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    source = _canonical(
        [start, start + timedelta(hours=1), start + timedelta(hours=3)],
        [100.0, 101.0, 110.0],
    )

    result = construct_horizon_targets(
        source,
        pd.DatetimeIndex([start]),
        horizon_hours=3,
        target_scope_end_exclusive=(start + timedelta(hours=10)).to_pydatetime(),
    )

    assert result.valid.empty
    assert result.counts["missing_endpoint_exclusions"] == 0
    assert result.counts["nonconsecutive_future_path_exclusions"] == 1
    assert result.counts["valid_targets"] == 0


def test_horizon_endpoint_cannot_cross_scope_boundary() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    times = [start + offset * timedelta(hours=1) for offset in range(5)]
    result = construct_horizon_targets(
        _canonical(times, [100.0, 101.0, 102.0, 103.0, 104.0]),
        pd.DatetimeIndex([start]),
        horizon_hours=3,
        target_scope_end_exclusive=(start + timedelta(hours=4)).to_pydatetime(),
    )

    assert result.valid.empty
    assert result.counts["train_boundary_crossing_exclusions"] == 1


def test_common_anchors_require_f0_and_all_four_targets() -> None:
    times = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    f0 = pd.DatetimeIndex(times[:4])
    targets = {
        1: pd.DatetimeIndex(times[:4]),
        3: pd.DatetimeIndex(times[1:5]),
        6: pd.DatetimeIndex(times[1:4]),
        12: pd.DatetimeIndex(times[2:4]),
    }

    first = common_anchor_times(f0, targets)
    second = common_anchor_times(f0, targets)

    assert first.equals(pd.DatetimeIndex(times[2:4]))
    assert _sample_identity(first) == _sample_identity(second)


def test_normalized_diagnostic_uses_exact_division_without_epsilon() -> None:
    result = normalized_diagnostic(
        np.array([1.0, -2.0]),
        np.array([2.0, 4.0]),
    )

    assert result["mean"] == 0.0
    assert result["rms"] == 0.5
    assert result["epsilon_added"] is False
    with pytest.raises(ValueError, match="strictly greater than zero"):
        normalized_diagnostic(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="finite targets and sigma"):
        normalized_diagnostic(np.array([1.0]), np.array([np.nan]))


def test_existing_target_read_is_physically_train_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return pd.DataFrame()

    def fake_read_table(path: Path, *, columns: list[str], filters: list[tuple]) -> FakeTable:
        captured.update(path=path, columns=columns, filters=filters)
        return FakeTable()

    monkeypatch.setattr(
        "btc_forecasting.targets.horizon_audit.pq.read_table",
        fake_read_table,
    )
    train_start = datetime(2020, 1, 1, tzinfo=UTC)
    validation_start = datetime(2021, 1, 1, tzinfo=UTC)
    _read_existing_train_targets(
        Path("targets.parquet"),
        train_start=train_start,
        validation_start=validation_start,
        target_end=validation_start,
    )

    assert captured["filters"] == [
        ("decision_time", ">=", train_start),
        ("decision_time", "<", validation_start),
        ("target_time", "<", validation_start),
    ]
