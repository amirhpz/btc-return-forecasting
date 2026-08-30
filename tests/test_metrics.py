import numpy as np

from btc_forecasting.evaluation.metrics import regression_metrics


def test_perfect_predictions_have_unit_skill_and_accuracy() -> None:
    truth = np.array([0.01, -0.02, 0.03])
    metrics = regression_metrics(truth, truth)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["mse_skill_vs_zero_return"] == 1.0


def test_zero_prediction_has_zero_mse_skill() -> None:
    truth = np.array([0.01, -0.02, 0.03])
    prediction = np.zeros_like(truth)
    metrics = regression_metrics(truth, prediction)
    assert np.isclose(metrics["mse_skill_vs_zero_return"], 0.0)
