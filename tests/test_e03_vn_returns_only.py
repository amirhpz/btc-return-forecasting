from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.training.lstm import LOOKBACK_HOURS, SequenceSamples, _sample_identity
from btc_forecasting.training.lstm_vn_mse import (
    prepare_volatility_normalized_samples,
    resolve_vn_mse_configuration,
)
from btc_forecasting.training.lstm_vn_returns_only import (
    EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY,
    RETURNS_FEATURE_INDEX,
    evaluate_returns_only_experiment,
    fit_returns_only_feature_scaler,
    project_returns_only,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
)

ROOT = Path(__file__).resolve().parents[1]


def _samples(
    normalized_targets: np.ndarray,
    sigma: np.ndarray,
    *,
    start: str,
) -> SequenceSamples:
    z = np.asarray(normalized_targets, dtype=np.float64)
    volatility = np.asarray(sigma, dtype=np.float64)
    generator = np.random.default_rng(42)
    features = generator.normal(
        size=(len(z), LOOKBACK_HOURS, len(F0_FEATURE_NAMES))
    ).astype(np.float32)
    features[:, -1, VOLATILITY_FEATURE_INDEX] = volatility
    return SequenceSamples(
        features=features,
        targets=z * volatility,
        decision_times=pd.date_range(start, periods=len(z), freq="h", tz="UTC"),
        candidate_count=len(z),
        excluded_lookback_count=0,
    )


def test_returns_projection_is_exactly_24_by_1_and_never_includes_sigma() -> None:
    prepared = prepare_volatility_normalized_samples(
        _samples(
            np.array([-1.0, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.03]),
            start="2026-01-01",
        )
    )
    expected_returns = prepared.normalized.features[:, :, RETURNS_FEATURE_INDEX].copy()

    projected = project_returns_only(prepared.normalized)

    assert projected.features.shape == (3, 24, 1)
    assert np.array_equal(projected.features[:, :, 0], expected_returns)
    assert projected.decision_times.equals(prepared.normalized.decision_times)
    assert np.array_equal(projected.targets, prepared.normalized.targets)
    assert not np.array_equal(projected.features[:, -1, 0], prepared.sigma)


def test_returns_scaler_is_fit_on_train_only_and_preserves_normalized_targets() -> None:
    train_full = prepare_volatility_normalized_samples(
        _samples(
            np.linspace(-1.0, 1.0, 8),
            np.linspace(0.01, 0.02, 8),
            start="2025-01-01",
        )
    )
    validation_full = prepare_volatility_normalized_samples(
        _samples(
            np.linspace(-1.0, 1.0, 4),
            np.linspace(0.02, 0.03, 4),
            start="2026-01-01",
        )
    )
    train = project_returns_only(train_full.normalized)
    validation = project_returns_only(validation_full.normalized)
    validation.features[:] = 1_000_000.0

    scaled = fit_returns_only_feature_scaler(train, validation)

    assert scaled.feature_scaler.n_features_in_ == 1
    assert scaled.feature_scaler.center_[0] == pytest.approx(
        np.median(train.features.reshape(-1))
    )
    assert np.array_equal(scaled.train.targets, train.targets)
    assert np.array_equal(scaled.validation.targets, validation.targets)


def test_r1_preserves_frozen_vn_mse_model_and_training_settings() -> None:
    resolved = resolve_vn_mse_configuration(project_root=ROOT)
    model = LSTMRegressor(
        input_size=1,
        hidden_size=int(resolved.model["hidden_size"]),
        num_layers=int(resolved.model["num_layers"]),
        configured_dropout=float(resolved.model["dropout"]),
    )

    assert model.input_size == 1
    assert model.hidden_size == 64
    assert model.num_layers == 1
    assert model.configured_dropout == 0.20
    assert model.effective_lstm_dropout == 0.0
    assert model(torch.zeros((3, 24, 1))).shape == (3,)
    assert resolved.training["loss"]["type"] == "torch.nn.MSELoss"
    assert resolved.training["optimizer"]["learning_rate"] == 0.001
    assert resolved.training["batch_size"] == 128
    assert resolved.training["max_epochs"] == 30
    assert resolved.training["early_stopping"]["patience"] == 5


def test_evaluation_uses_raw_reconstruction_and_reports_normalized_metrics() -> None:
    train = prepare_volatility_normalized_samples(
        _samples(
            np.linspace(-1.0, 1.0, 8),
            np.linspace(0.01, 0.02, 8),
            start="2025-01-01",
        )
    )
    validation = prepare_volatility_normalized_samples(
        _samples(
            np.array([-1.0, -0.5, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.03, 0.04]),
            start="2026-01-01",
        )
    )
    predictions = validation.normalized.targets * 0.5
    identity = _sample_identity(validation.raw.decision_times)

    result = evaluate_returns_only_experiment(
        train=train,
        validation=validation,
        normalized_predictions=predictions,
        expected_validation_identity=identity,
    )

    expected_raw = predictions * validation.sigma
    assert result["experiment_id"] == "E03-VN-R1"
    assert result["lstm_validation"]["mae"] == pytest.approx(
        np.mean(np.abs(validation.raw.targets - expected_raw))
    )
    assert result["normalized_space_validation"]["mse"] == pytest.approx(
        np.mean(np.square(validation.normalized.targets - predictions))
    )
    assert result["normalized_space_validation"][
        "prediction_std_over_target_std"
    ] == pytest.approx(0.5)
    assert result["comparability_to_e03_vn_mse"] == {
        "expected_sample_identity_sha256": identity,
        "actual_sample_identity_sha256": identity,
        "matches": True,
        "difference_reason": None,
    }
    assert result["data"]["window_shape"] == [24, 1]
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_identity_difference_reports_exact_invalid_sigma_exclusion() -> None:
    train = prepare_volatility_normalized_samples(
        _samples(np.linspace(-1.0, 1.0, 8), np.ones(8), start="2025-01-01")
    )
    source_validation = _samples(
        np.linspace(-1.0, 1.0, 5),
        np.ones(5),
        start="2026-01-01",
    )
    expected_identity = _sample_identity(source_validation.decision_times)
    source_validation.features[2, -1, VOLATILITY_FEATURE_INDEX] = 0.0
    validation = prepare_volatility_normalized_samples(source_validation)

    result = evaluate_returns_only_experiment(
        train=train,
        validation=validation,
        normalized_predictions=np.zeros(4),
        expected_validation_identity=expected_identity,
    )

    comparison = result["comparability_to_e03_vn_mse"]
    assert comparison["matches"] is False
    assert "1 validation samples were excluded" in comparison["difference_reason"]


def test_r1_identity_and_cli_are_distinct_and_frozen() -> None:
    assert EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY == (
        "3907f80c1b59c98d13d1a733953e7a679b528747b953d3e9db58c63cc10ba13c"
    )
    args = build_parser().parse_args(["run-lstm-vn-returns-only"])
    assert args.command == "run-lstm-vn-returns-only"
