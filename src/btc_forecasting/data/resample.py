from __future__ import annotations

import pandas as pd

from btc_forecasting.data.schema import ADDITIVE_COLUMNS


def resample_5m_to_1h(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "open_time",
    expected_source_bars: int = 12,
    require_complete_buckets: bool = True,
) -> pd.DataFrame:
    """Resample complete 5-minute OHLCV bars into deterministic 1-hour bars."""
    work = frame.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="raise")
    work = work.sort_values(timestamp_col).set_index(timestamp_col)

    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    for column in ADDITIVE_COLUMNS:
        if column in work.columns:
            aggregation[column] = "sum"

    grouped = work.resample("1h", label="left", closed="left")
    result = grouped.agg(aggregation)
    counts = grouped["close"].count().rename("source_bar_count")
    result = result.join(counts)

    if require_complete_buckets:
        result = result.loc[result["source_bar_count"] == expected_source_bars]

    result = result.dropna(subset=["open", "high", "low", "close"])
    return result.reset_index()
