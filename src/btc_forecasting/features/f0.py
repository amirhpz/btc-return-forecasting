from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

import numpy as np
import pandas as pd

F0_FEATURE_NAMES = (
    "log_return_1h",
    "log_return_3h",
    "log_return_6h",
    "candle_log_return",
    "high_low_range_pct",
    "close_position_in_range",
    "rolling_volatility_6h",
    "rolling_volatility_24h",
    "log_volume_change_1h",
    "relative_volume_24h",
)
REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")
ONE_HOUR = timedelta(hours=1)


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required F0 columns: {missing}")


def _consecutive_history_mask(open_time: pd.Series, lag_hours: int) -> pd.Series:
    """Return true only when the preceding rows cover exact consecutive UTC hours."""
    return open_time.sub(open_time.shift(lag_hours)).eq(lag_hours * ONE_HOUR)


def _validated_hourly_input(hourly: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    _require_columns(hourly, REQUIRED_COLUMNS)
    open_time = pd.to_datetime(hourly["open_time"], utc=True, errors="coerce")
    if open_time.isna().any():
        raise ValueError("open_time must contain valid UTC timestamps")
    if open_time.duplicated().any() or not open_time.is_monotonic_increasing:
        raise ValueError("open_time must be strictly ordered and unique")
    if not open_time.dt.floor("h").eq(open_time).all():
        raise ValueError("open_time must lie on UTC hour boundaries")

    numeric = hourly[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("F0 source columns must contain finite numeric values")
    if (numeric[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("F0 price columns must be strictly positive")
    if (numeric["volume"] < 0.0).any():
        raise ValueError("F0 volume must be non-negative")
    return open_time, numeric


def compute_f0_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen causal F0 feature set from canonical hourly bars.

    A row with ``open_time=t`` uses only the completed bar at ``t`` and earlier
    consecutive hourly bars. It becomes available at ``decision_time=t+1h``.
    Insufficient or gap-crossing histories remain missing.
    """
    open_time, values = _validated_hourly_input(hourly)
    open_price = values["open"]
    high = values["high"]
    low = values["low"]
    close = values["close"]
    volume = values["volume"]

    consecutive_1h = _consecutive_history_mask(open_time, 1)
    log_close = np.log(close)
    log_return_1h = log_close.sub(log_close.shift(1)).where(consecutive_1h)

    candle_range = high - low
    close_position = (close - low).div(candle_range)
    close_position = close_position.mask(candle_range.eq(0.0), 0.5)

    previous_24h_mean_volume = volume.shift(1).rolling(
        window=24, min_periods=24
    ).mean()
    relative_volume = volume.div(previous_24h_mean_volume).where(
        previous_24h_mean_volume.ne(0.0)
        & _consecutive_history_mask(open_time, 24)
    )

    result = pd.DataFrame(index=hourly.index)
    result["open_time"] = open_time
    result["decision_time"] = open_time + ONE_HOUR
    result["log_return_1h"] = log_return_1h
    result["log_return_3h"] = log_close.sub(log_close.shift(3)).where(
        _consecutive_history_mask(open_time, 3)
    )
    result["log_return_6h"] = log_close.sub(log_close.shift(6)).where(
        _consecutive_history_mask(open_time, 6)
    )
    result["candle_log_return"] = np.log(close.div(open_price))
    result["high_low_range_pct"] = candle_range.div(close)
    result["close_position_in_range"] = close_position
    result["rolling_volatility_6h"] = log_return_1h.rolling(
        window=6, min_periods=6
    ).std(ddof=0)
    result["rolling_volatility_24h"] = log_return_1h.rolling(
        window=24, min_periods=24
    ).std(ddof=0)
    result["log_volume_change_1h"] = np.log1p(volume).sub(
        np.log1p(volume.shift(1))
    ).where(consecutive_1h)
    result["relative_volume_24h"] = relative_volume
    return result
