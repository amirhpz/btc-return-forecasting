import numpy as np
import pandas as pd

from btc_forecasting.targets.future_return import future_log_return, target_timestamps


def test_future_return_and_timestamp_alignment() -> None:
    close = pd.Series([100.0, 110.0, 121.0])
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"))
    target = future_log_return(close, horizon_bars=1)
    future_times = target_timestamps(timestamps, horizon_bars=1)
    assert np.isclose(target.iloc[0], np.log(1.1))
    assert np.isclose(target.iloc[1], np.log(1.1))
    assert pd.isna(target.iloc[2])
    assert future_times.iloc[0] == timestamps.iloc[1]
    assert pd.isna(future_times.iloc[2])
