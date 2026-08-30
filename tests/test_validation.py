import pandas as pd

from btc_forecasting.data.validation import validate_ohlcv


def test_validation_accepts_clean_frame() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1.0, 2.0, 3.0],
        }
    )
    report = validate_ohlcv(frame)
    assert report.duplicate_timestamps == 0
    assert report.missing_interval_count == 0
    assert report.invalid_ohlc_rows == 0
