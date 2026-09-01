from __future__ import annotations

from datetime import timedelta
import warnings

import numpy as np
import pandas as pd

from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES, compute_eng52_features
from btc_forecasting.features.eng52_qc import _coverage, _numerical_statistics


def _source(periods: int = 80) -> pd.DataFrame:
    position = np.arange(periods, dtype=float)
    close = 10_000.0 + 4.0 * position + 30.0 * np.sin(position / 4.0)
    open_price = close * (1.0 + 0.001 * np.cos(position / 3.0))
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2022-01-01", periods=periods, freq="h", tz="UTC"),
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.003,
            "low": np.minimum(open_price, close) * 0.997,
            "close": close,
            "volume": 50.0 + position % 17,
        }
    )


def test_expected_gap_warmup_nans_do_not_emit_all_nan_warning() -> None:
    source = _source().drop(index=[40, 75]).reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_eng52_features(source)

    assert not any("All-NaN slice encountered" in str(item.message) for item in caught)
    assert result["lower_wick_to_tr"].iloc[40] != result["lower_wick_to_tr"].iloc[40]
    assert result["upper_wick_to_tr"].iloc[40] != result["upper_wick_to_tr"].iloc[40]


def test_numerical_qc_reports_required_fields_and_inf() -> None:
    values = np.linspace(-1.0, 1.0, 20)
    frame = pd.DataFrame({name: values.copy() for name in ENG52_FEATURE_NAMES})
    frame.loc[0, ENG52_FEATURE_NAMES[0]] = np.inf
    statistics, flags = _numerical_statistics(frame)

    assert statistics[ENG52_FEATURE_NAMES[0]]["inf_count"] == 1  # type: ignore[index]
    assert flags["total_inf_count"] == 1
    assert set(statistics[ENG52_FEATURE_NAMES[1]]) >= {  # type: ignore[arg-type]
        "finite_count", "missing_count", "unique_finite_value_count", "q001",
        "q999", "zero_ratio_among_finite", "flags",
    }


def test_coverage_separates_window_and_nonfinite_exclusions() -> None:
    times = pd.date_range("2023-01-01", periods=28, freq="h", tz="UTC")
    feature_rows = pd.DataFrame({"open_time": times, "x": np.arange(28, dtype=float)})
    feature_rows.loc[25, "x"] = np.nan
    anchors = [times[10], times[23], times[25], times[27]]
    targets = pd.DataFrame(
        {
            "bar_open_time": anchors,
            "decision_time": [value.to_pydatetime() + timedelta(hours=1) for value in anchors],
        }
    )

    result = _coverage(feature_rows, ("x",), targets)

    assert result["candidate_target_rows"] == 4
    assert result["usable_samples"] == 1
    assert result["excluded_incomplete_or_nonconsecutive_24h_window"] == 1
    assert result["excluded_missing_or_nonfinite_features"] == 2
