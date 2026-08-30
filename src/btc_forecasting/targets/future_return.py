from __future__ import annotations

import numpy as np
import pandas as pd


def future_log_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Return log(close[t+h] / close[t]); final h rows are missing by construction."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    numeric_close = pd.to_numeric(close, errors="raise")
    return np.log(numeric_close.shift(-horizon_bars) / numeric_close).rename(
        f"future_log_return_h{horizon_bars}"
    )


def target_timestamps(timestamps: pd.Series, horizon_bars: int) -> pd.Series:
    """Return the timestamp associated with each future target."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    parsed = pd.to_datetime(timestamps, utc=True, errors="raise")
    return parsed.shift(-horizon_bars).rename("target_timestamp")
