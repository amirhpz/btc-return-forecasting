from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset

from btc_forecasting.data.schema import PRICE_COLUMNS, REQUIRED_OHLCV_COLUMNS


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    duplicate_timestamps: int
    non_monotonic_timestamps: bool
    missing_interval_count: int
    nonpositive_price_rows: int
    negative_volume_rows: int
    zero_volume_rows: int
    invalid_ohlc_rows: int

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "open_time",
    expected_interval: str = "5min",
) -> ValidationReport:
    """Validate an OHLCV frame without mutating or repairing it."""
    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="raise")
    duplicate_timestamps = int(timestamps.duplicated().sum())
    non_monotonic = not timestamps.is_monotonic_increasing

    ordered_unique = timestamps.drop_duplicates().sort_values()
    expected_delta = pd.Timedelta(to_offset(expected_interval).nanos, unit="ns")
    diffs = ordered_unique.diff().dropna()
    missing_intervals = int(((diffs / expected_delta) - 1).clip(lower=0).sum())

    prices = frame.loc[:, list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    nonpositive_price_rows = int((prices <= 0).any(axis=1).sum())
    negative_volume_rows = int((volume < 0).sum())
    zero_volume_rows = int((volume == 0).sum())

    invalid_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
    )

    return ValidationReport(
        row_count=len(frame),
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic,
        missing_interval_count=missing_intervals,
        nonpositive_price_rows=nonpositive_price_rows,
        negative_volume_rows=negative_volume_rows,
        zero_volume_rows=zero_volume_rows,
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
    )
