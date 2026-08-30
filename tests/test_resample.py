import pandas as pd

from btc_forecasting.data.resample import resample_5m_to_1h


def test_resample_requires_twelve_complete_bars() -> None:
    timestamps = pd.date_range("2026-01-01 00:00", periods=13, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open_time": timestamps,
            "open": range(100, 113),
            "high": range(101, 114),
            "low": range(99, 112),
            "close": range(100, 113),
            "volume": [1.0] * 13,
        }
    )
    result = resample_5m_to_1h(frame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["open"] == 100
    assert row["high"] == 112
    assert row["low"] == 99
    assert row["close"] == 111
    assert row["volume"] == 12
    assert row["source_bar_count"] == 12
