from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.baselines.ridge import (
    LOOKBACK_HOURS,
    RidgeSamples,
    _load_resolved_configuration,
    _sample_identity,
    fit_ridge_model,
)
from btc_forecasting.baselines.ridge_vn import (
    ENDPOINT_VOLATILITY_COLUMN,
    EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY,
    evaluate_vn_ridge_control,
    prepare_volatility_normalized_ridge_samples,
)
from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _samples(
    normalized_targets: np.ndarray,
    sigma: np.ndarray,
    *,
    start: str,
) -> RidgeSamples:
    z = np.asarray(normalized_targets, dtype=np.float64)
    volatility = np.asarray(sigma, dtype=np.float64)
    generator = np.random.default_rng(42)
    features = generator.normal(
        size=(len(z), LOOKBACK_HOURS * len(F0_FEATURE_NAMES))
    )
    features[:, ENDPOINT_VOLATILITY_COLUMN] = volatility
    return RidgeSamples(
        features=features,
        targets=z * volatility,
        decision_times=pd.date_range(start, periods=len(z), freq="h", tz="UTC"),
        candidate_count=len(z),
        excluded_lookback_count=0,
    )


def test_ridge_normalization_uses_only_flattened_sequence_endpoint_without_epsilon() -> None:
    samples = _samples(
        np.array([2.0, -3.0, 1.0]),
        np.array([0.01, 0.02, 0.03]),
        start="2026-01-01",
    )
    earlier_volatility_column = ENDPOINT_VOLATILITY_COLUMN - len(F0_FEATURE_NAMES)
    samples.features[:, earlier_volatility_column] = 999.0

    prepared = prepare_volatility_normalized_ridge_samples(samples)

    assert prepared.exclusion_count == 0
    assert prepared.sigma.tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert prepared.normalized.targets.tolist() == pytest.approx([2.0, -3.0, 1.0])
    assert np.all(samples.features[:, earlier_volatility_column] == 999.0)


def test_invalid_endpoint_sigma_is_excluded_exactly() -> None:
    samples = _samples(
        np.arange(1.0, 7.0),
        np.ones(6),
        start="2026-01-01",
    )
    samples.features[:, ENDPOINT_VOLATILITY_COLUMN] = np.array(
        [2.0, 0.0, np.nan, np.inf, -1.0, 3.0]
    )

    prepared = prepare_volatility_normalized_ridge_samples(samples)

    assert prepared.original_eligible_count == 6
    assert prepared.exclusion_count == 4
    assert prepared.raw.targets.tolist() == [1.0, 6.0]
    assert prepared.normalized.targets.tolist() == pytest.approx([0.5, 2.0])
    assert prepared.raw.decision_times.equals(samples.decision_times[[0, 5]])


def test_control_preserves_frozen_e02_ridge_and_train_only_scalers() -> None:
    _, model_config = _load_resolved_configuration(ROOT)
    train = prepare_volatility_normalized_ridge_samples(
        _samples(
            np.linspace(-1.0, 1.0, 12),
            np.linspace(0.01, 0.03, 12),
            start="2025-01-01",
        )
    )

    trained = fit_ridge_model(
        train.normalized,
        alpha=float(model_config["alpha"]),
        fit_intercept=bool(model_config["fit_intercept"]),
    )

    assert trained.model.alpha == 1.0
    assert trained.model.fit_intercept is True
    assert trained.feature_scaler.n_features_in_ == 24 * 10
    assert trained.target_scaler.mean_[0] == pytest.approx(
        np.mean(train.normalized.targets)
    )


def test_validation_reconstructs_raw_returns_and_reports_both_spaces_without_test() -> None:
    train = prepare_volatility_normalized_ridge_samples(
        _samples(
            np.linspace(-1.0, 1.0, 20),
            np.linspace(0.01, 0.03, 20),
            start="2025-01-01",
        )
    )
    validation = prepare_volatility_normalized_ridge_samples(
        _samples(
            np.array([-1.0, -0.5, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.03, 0.04]),
            start="2026-01-01",
        )
    )
    trained = fit_ridge_model(train.normalized, alpha=1.0, fit_intercept=True)
    identity = _sample_identity(validation.raw.decision_times)

    evaluation = evaluate_vn_ridge_control(
        trained,
        train=train,
        validation=validation,
        expected_validation_identity=identity,
    )

    assert evaluation.raw_predictions.tolist() == pytest.approx(
        evaluation.normalized_predictions * validation.sigma
    )
    assert evaluation.result["raw_return_validation"]["n"] == 4
    assert evaluation.result["normalized_space_validation"]["mse"] == pytest.approx(
        np.mean(
            np.square(
                validation.normalized.targets - evaluation.normalized_predictions
            )
        )
    )
    assert evaluation.result["zero_return_same_validation_rows"]["n"] == 4
    assert evaluation.result["e03_vn_mse_sample_identity"] == {
        "expected": identity,
        "actual": identity,
        "matches": True,
    }
    assert evaluation.result["evaluated_splits"] == ["validation"]
    assert evaluation.result["test_set"] == "NOT EVALUATED"
    assert "test" not in evaluation.result


def test_identity_guard_and_distinct_cli_are_frozen() -> None:
    assert EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY == (
        "3907f80c1b59c98d13d1a733953e7a679b528747b953d3e9db58c63cc10ba13c"
    )
    train = prepare_volatility_normalized_ridge_samples(
        _samples(np.linspace(-1.0, 1.0, 8), np.ones(8), start="2025-01-01")
    )
    validation = prepare_volatility_normalized_ridge_samples(
        _samples(np.linspace(-1.0, 1.0, 4), np.ones(4), start="2026-01-01")
    )
    trained = fit_ridge_model(train.normalized, alpha=1.0, fit_intercept=True)

    with pytest.raises(ValueError, match="validation identity differs"):
        evaluate_vn_ridge_control(
            trained,
            train=train,
            validation=validation,
        )

    args = build_parser().parse_args(["run-vn-ridge-control"])
    assert args.command == "run-vn-ridge-control"
