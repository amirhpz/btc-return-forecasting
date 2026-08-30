from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from btc_forecasting.splits.frozen import create_frozen_chronological_split


def _source(decision_times: list[datetime]) -> pa.Table:
    return pa.table(
        {
            "decision_time": pa.array(
                decision_times,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "target_time": pa.array(
                [value + timedelta(hours=1) for value in decision_times],
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )


def test_chronological_seventy_fifteen_fifteen_split() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    decision_times = [start + timedelta(hours=index) for index in range(14)]
    decision_times.extend(start + timedelta(hours=index) for index in range(16, 19))
    decision_times.extend(start + timedelta(hours=index) for index in range(21, 24))

    result = create_frozen_chronological_split(_source(list(reversed(decision_times))))
    split = result.metadata["split"]

    assert split["provisional_rows"] == {"train": 14, "validation": 3, "test": 3}
    assert split["retained_rows"] == {"train": 14, "validation": 3, "test": 3}
    assert split["boundaries"]["train"]["decision_time_end_exclusive"] == (
        "2026-01-01T16:00:00Z"
    )
    assert split["boundaries"]["validation"]["decision_time_end_exclusive"] == (
        "2026-01-01T21:00:00Z"
    )
    assert split["invariants"]["chronologically_ordered"] is True


def test_split_assignments_have_zero_overlap() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    result = create_frozen_chronological_split(
        _source([start + timedelta(hours=index) for index in range(20)])
    )

    train = set(result.train_indices)
    validation = set(result.validation_indices)
    test = set(result.test_indices)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert result.metadata["split"]["invariants"]["zero_overlap"] is True


def test_targets_crossing_train_and_validation_boundaries_are_purged() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    result = create_frozen_chronological_split(
        _source([start + timedelta(hours=index) for index in range(20)])
    )
    split = result.metadata["split"]

    assert split["provisional_rows"] == {"train": 14, "validation": 3, "test": 3}
    assert split["retained_rows"] == {"train": 13, "validation": 2, "test": 3}
    assert split["purged_boundary_rows"] == {"train": 1, "validation": 1, "total": 2}
    assert split["invariants"]["no_target_crosses_split_boundary"] is True
