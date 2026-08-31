from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from btc_forecasting.baselines.ridge import (
    RidgeSamples,
    build_lookback_samples,
    evaluate_ridge_baseline,
    fit_ridge_model,
)
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.targets.one_hour import TARGET_COLUMN


def _samples(features: np.ndarray, targets: np.ndarray, *, start: str) -> RidgeSamples:
    return RidgeSamples(
        features=np.asarray(features, dtype=float),
        targets=np.asarray(targets, dtype=float),
        decision_times=pd.date_range(start, periods=len(targets), freq="h", tz="UTC"),
        candidate_count=len(targets),
        excluded_lookback_count=0,
    )


def test_lookback_windows_never_cross_an_hourly_gap() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    hours = [hour for hour in range(35) if hour != 10]
    open_times = pd.DatetimeIndex([start + timedelta(hours=hour) for hour in hours])
    feature_rows = pd.DataFrame({"open_time": open_times})
    for feature_index, name in enumerate(F0_FEATURE_NAMES):
        feature_rows[name] = np.arange(len(feature_rows), dtype=float) + feature_index
    anchors = pd.DatetimeIndex([start + timedelta(hours=24), start + timedelta(hours=34)])
    targets = pd.DataFrame(
        {
            "bar_open_time": anchors,
            "decision_time": anchors + timedelta(hours=1),
            "target_time": anchors + timedelta(hours=2),
            TARGET_COLUMN: [0.1, 0.2],
        }
    )

    samples = build_lookback_samples(feature_rows, targets)

    assert samples.candidate_count == 2
    assert samples.excluded_lookback_count == 1
    assert samples.features.shape == (1, 24 * 10)
    assert samples.targets.tolist() == [0.2]
    assert samples.decision_times.tolist() == [start + timedelta(hours=35)]


def test_feature_and_target_scalers_fit_train_only() -> None:
    train = _samples(
        np.array([[0.0, 10.0], [2.0, 20.0], [100.0, 30.0]]),
        np.array([1.0, 2.0, 9.0]),
        start="2026-01-01",
    )
    validation = _samples(
        np.array([[1_000_000.0, 2_000_000.0], [3_000_000.0, 4_000_000.0]]),
        np.array([100_000.0, 200_000.0]),
        start="2026-02-01",
    )

    trained = fit_ridge_model(train, alpha=1.0, fit_intercept=True)
    evaluate_ridge_baseline(trained, train=train, validation=validation)

    assert np.array_equal(trained.feature_scaler.center_, np.array([2.0, 20.0]))
    assert np.array_equal(trained.target_scaler.mean_, np.array([4.0]))


def test_ridge_and_zero_use_identical_validation_rows_without_test_metrics() -> None:
    train = _samples(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
                [4.0, 1.0],
            ]
        ),
        np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        start="2026-01-01",
    )
    validation = _samples(
        np.array([[5.0, 0.0], [6.0, 1.0], [7.0, 0.0]]),
        np.array([-0.3, 0.1, 0.4]),
        start="2026-02-01",
    )
    trained = fit_ridge_model(train, alpha=1.0, fit_intercept=True)

    evaluation = evaluate_ridge_baseline(trained, train=train, validation=validation)
    result = evaluation.result

    ridge_validation = result["ridge"]["validation"]
    zero_validation = result["zero_return_same_validation_rows"]
    assert ridge_validation["n"] == zero_validation["n"] == 3
    assert ridge_validation["sample_identity_sha256"] == zero_validation[
        "sample_identity_sha256"
    ]
    assert result["evaluated_splits"] == ["train", "validation"]
    assert "test" not in result["ridge"]
    assert result["test_set"] == "NOT EVALUATED"
