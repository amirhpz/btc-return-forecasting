from __future__ import annotations

from btc_forecasting.training.lstm_vn_multiseed import PRIMARY_METRIC_PATHS
from btc_forecasting.training.lstm_vn_walkforward import temporal_stability_gate


def _aggregate(*, regression_positive: bool) -> dict[str, dict[str, float]]:
    value = 0.01 if regression_positive else -0.01
    aggregate = {
        name: {
            "mean": value,
            "median": value,
            "standard_deviation": 0.0,
            "minimum": value,
            "maximum": value,
        }
        for name in PRIMARY_METRIC_PATHS
    }
    directional_accuracy = 0.51 if regression_positive else 0.49
    aggregate["directional_accuracy"] = {
        "mean": directional_accuracy,
        "median": directional_accuracy,
        "standard_deviation": 0.0,
        "minimum": directional_accuracy,
        "maximum": directional_accuracy,
    }
    return aggregate


def test_horizon_regression_stability_gate_is_the_exact_frozen_wf4_rule() -> None:
    fold_aggregates = [_aggregate(regression_positive=True) for _ in range(3)] + [
        _aggregate(regression_positive=False)
    ]
    positive, stable = temporal_stability_gate(
        fold_aggregates,
        _aggregate(regression_positive=True),
    )
    assert positive == [True, True, True, False]
    assert stable is True

    _, unstable_two_folds = temporal_stability_gate(
        [_aggregate(regression_positive=True) for _ in range(2)]
        + [_aggregate(regression_positive=False) for _ in range(2)],
        _aggregate(regression_positive=True),
    )
    assert unstable_two_folds is False

    overall_failure = _aggregate(regression_positive=True)
    overall_failure["rmse_skill"]["mean"] = 0.0
    _, unstable_overall = temporal_stability_gate(fold_aggregates, overall_failure)
    assert unstable_overall is False
