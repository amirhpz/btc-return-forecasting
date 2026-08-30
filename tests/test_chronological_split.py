import pandas as pd

from btc_forecasting.splits.chronological import assign_splits_by_target_timestamp


def test_split_is_monotonic_and_complete_for_valid_targets() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC"))
    labels = assign_splits_by_target_timestamp(timestamps)
    assert set(labels.dropna().unique()) == {"train", "validation", "test"}
    encoded = labels.map({"train": 0, "validation": 1, "test": 2})
    assert encoded.is_monotonic_increasing
