from __future__ import annotations

import math

import numpy as np

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.baselines.zero_return import evaluate_zero_return_baseline
from btc_forecasting.evaluation.metrics import regression_loss_metrics


def test_zero_return_predictions_are_exactly_zero() -> None:
    predictions = zero_return_prediction(5)

    assert predictions.dtype == np.float64
    assert np.array_equal(predictions, np.zeros(5, dtype=np.float64))


def test_zero_return_mae_and_rmse_are_calculated_correctly() -> None:
    truth = np.array([1.0, -1.0])

    metrics = regression_loss_metrics(truth, zero_return_prediction(len(truth)))

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["r2"] == 0.0


def test_only_train_and_validation_are_evaluated() -> None:
    result = evaluate_zero_return_baseline(
        train_targets=np.array([0.1, -0.1]),
        validation_targets=np.array([0.2, -0.2]),
    )

    assert result["evaluated_splits"] == ["train", "validation"]
    assert set(result["metrics"]) == {"train", "validation"}
    assert result["metrics"]["train"]["n"] == 2
    assert result["metrics"]["validation"]["n"] == 2


def test_test_metrics_are_not_produced() -> None:
    result = evaluate_zero_return_baseline(
        train_targets=np.array([0.1, -0.1]),
        validation_targets=np.array([0.2, -0.2]),
    )

    assert "test" not in result["metrics"]
    assert "test" not in result["reference_losses"]
    assert result["test_set"] == "NOT EVALUATED"
    assert math.isnan(result["metrics"]["train"]["r2"]) is False
