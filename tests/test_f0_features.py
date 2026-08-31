from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd

from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features


def _hourly_frame(
    count: int,
    *,
    omitted_hour: int | None = None,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    hours = [hour for hour in range(count) if hour != omitted_hour]
    positions = np.asarray(hours, dtype=float)
    close = np.exp(4.0 + 0.01 * positions + 0.0001 * positions**2)
    if volumes is None:
        volume = positions + 1.0
    else:
        volume = np.asarray(volumes, dtype=float)
        if len(volume) != len(hours):
            raise ValueError("volumes must match the retained hourly rows")
    return pd.DataFrame(
        {
            "open_time": pd.Timestamp("2026-01-01T00:00:00Z")
            + pd.to_timedelta(positions, unit="h"),
            "open": close / np.exp(0.002),
            "high": close * 1.03,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
        }
    )


def test_f0_exact_formulas_alignment_and_count() -> None:
    source = _hourly_frame(30)
    result = compute_f0_features(source)
    row = 24
    close = source["close"].to_numpy()
    volume = source["volume"].to_numpy()

    assert len(F0_FEATURE_NAMES) == 10
    assert tuple(result.columns[2:]) == F0_FEATURE_NAMES
    assert result.loc[row, "decision_time"] == source.loc[row, "open_time"] + timedelta(
        hours=1
    )
    assert math.isclose(result.loc[row, "log_return_1h"], math.log(close[row] / close[row - 1]))
    assert math.isclose(result.loc[row, "log_return_3h"], math.log(close[row] / close[row - 3]))
    assert math.isclose(result.loc[row, "log_return_6h"], math.log(close[row] / close[row - 6]))
    assert math.isclose(
        result.loc[row, "candle_log_return"],
        math.log(source.loc[row, "close"] / source.loc[row, "open"]),
    )
    assert math.isclose(
        result.loc[row, "high_low_range_pct"],
        (source.loc[row, "high"] - source.loc[row, "low"]) / source.loc[row, "close"],
    )
    assert math.isclose(result.loc[row, "close_position_in_range"], 0.4)

    last_6_returns = np.log(close[row - 5 : row + 1] / close[row - 6 : row])
    last_24_returns = np.log(close[1 : row + 1] / close[:row])
    assert math.isclose(
        result.loc[row, "rolling_volatility_6h"], np.std(last_6_returns, ddof=0)
    )
    assert math.isclose(
        result.loc[row, "rolling_volatility_24h"], np.std(last_24_returns, ddof=0)
    )
    assert math.isclose(
        result.loc[row, "log_volume_change_1h"],
        math.log1p(volume[row]) - math.log1p(volume[row - 1]),
    )
    assert math.isclose(
        result.loc[row, "relative_volume_24h"], volume[row] / np.mean(volume[:row])
    )


def test_f0_is_prefix_causal_without_future_leakage() -> None:
    source = _hourly_frame(32)
    cutoff = 25
    original = compute_f0_features(source)
    perturbed = source.copy()
    future = perturbed.index > cutoff
    perturbed.loc[future, ["open", "high", "low", "close", "volume"]] *= 10.0

    changed = compute_f0_features(perturbed)

    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_lagged_and_rolling_features_never_cross_an_hourly_gap() -> None:
    source = _hourly_frame(40, omitted_hour=10)
    result = compute_f0_features(source).set_index("open_time")
    gap_successor = pd.Timestamp("2026-01-01T11:00:00Z")
    lagged = [
        "log_return_1h",
        "log_return_3h",
        "log_return_6h",
        "rolling_volatility_6h",
        "rolling_volatility_24h",
        "log_volume_change_1h",
        "relative_volume_24h",
    ]

    assert result.loc[gap_successor, lagged].isna().all()
    assert math.isfinite(result.loc[pd.Timestamp("2026-01-01T17:00:00Z"), "rolling_volatility_6h"])
    assert pd.isna(result.loc[pd.Timestamp("2026-01-02T10:00:00Z"), "rolling_volatility_24h"])
    assert math.isfinite(
        result.loc[pd.Timestamp("2026-01-02T11:00:00Z"), "rolling_volatility_24h"]
    )
    assert math.isfinite(
        result.loc[pd.Timestamp("2026-01-02T11:00:00Z"), "relative_volume_24h"]
    )


def test_flat_range_close_position_is_exactly_one_half() -> None:
    source = _hourly_frame(1)
    source.loc[0, ["high", "low", "close"]] = 100.0

    result = compute_f0_features(source)

    assert result.loc[0, "close_position_in_range"] == 0.5


def test_log_volume_change_uses_log1p_and_accepts_zero_volume() -> None:
    source = _hourly_frame(3, volumes=np.asarray([0.0, math.e**2 - 1.0, 0.0]))

    result = compute_f0_features(source)

    assert math.isclose(result.loc[1, "log_volume_change_1h"], 2.0)
    assert math.isclose(result.loc[2, "log_volume_change_1h"], -2.0)


def test_relative_volume_is_missing_for_exactly_zero_denominator() -> None:
    volume = np.zeros(26)
    volume[24] = 12.0
    volume[25] = 6.0
    source = _hourly_frame(26, volumes=volume)

    result = compute_f0_features(source)

    assert pd.isna(result.loc[24, "relative_volume_24h"])
    assert result.loc[25, "relative_volume_24h"] == 12.0
