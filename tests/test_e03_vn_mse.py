from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_mse import (
    _changed_leaf_paths,
    resolve_mse_configuration,
)
from btc_forecasting.training.lstm_vn_mse import (
    NORMALIZED_TARGET_SCALE,
    evaluate_vn_mse_experiment,
    normalized_space_diagnostics,
    prepare_volatility_normalized_samples,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
)

ROOT = Path(__file__).resolve().parents[1]


def _samples(
    targets: np.ndarray,
    endpoint_sigma: np.ndarray,
    *,
    start: str,
) -> SequenceSamples:
    truth = np.asarray(targets, dtype=np.float64)
    sigma = np.asarray(endpoint_sigma, dtype=np.float64)
    features = np.zeros((len(truth), 24, len(F0_FEATURE_NAMES)), dtype=np.float32)
    features[:, :, VOLATILITY_FEATURE_INDEX] = 999.0
    features[:, -1, VOLATILITY_FEATURE_INDEX] = sigma
    return SequenceSamples(
        features=features,
        targets=truth,
        decision_times=pd.date_range(start, periods=len(truth), freq="h", tz="UTC"),
        candidate_count=len(truth),
        excluded_lookback_count=0,
    )


def test_target_normalization_uses_only_causal_endpoint_sigma_without_epsilon() -> None:
    samples = _samples(
        np.array([0.02, -0.06, 0.03]),
        np.array([0.01, 0.02, 0.03]),
        start="2026-01-01",
    )
    before = samples.targets.copy()

    prepared = prepare_volatility_normalized_samples(samples)

    assert prepared.exclusion_count == 0
    assert prepared.sigma.tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert prepared.normalized.targets.tolist() == pytest.approx([2.0, -3.0, 1.0])
    assert np.array_equal(samples.targets, before)
    assert np.all(samples.features[:, 0, VOLATILITY_FEATURE_INDEX] == 999.0)


def test_invalid_sigma_is_excluded_exactly() -> None:
    samples = _samples(
        np.arange(1.0, 7.0),
        np.array([2.0, 0.0, np.nan, np.inf, -1.0, 3.0]),
        start="2026-01-01",
    )

    prepared = prepare_volatility_normalized_samples(samples)

    assert prepared.original_eligible_count == 6
    assert prepared.exclusion_count == 4
    assert prepared.raw.targets.tolist() == [1.0, 6.0]
    assert prepared.normalized.targets.tolist() == pytest.approx([0.5, 2.0])
    assert prepared.raw.decision_times.equals(samples.decision_times[[0, 5]])


def test_raw_prediction_reconstruction_is_exact_and_rejects_invalid_sigma() -> None:
    reconstructed = reconstruct_raw_predictions(
        np.array([2.0, -3.0, 0.5]),
        np.array([0.01, 0.02, 0.04]),
    )

    assert reconstructed.tolist() == pytest.approx([0.02, -0.06, 0.02])
    with pytest.raises(ValueError, match="finite positive sigma"):
        reconstruct_raw_predictions(np.array([1.0]), np.array([0.0]))


def test_evaluation_uses_reconstructed_raw_returns_and_never_test() -> None:
    train = prepare_volatility_normalized_samples(
        _samples(
            np.array([0.01, -0.02, 0.03]),
            np.array([0.01, 0.02, 0.03]),
            start="2025-01-01",
        )
    )
    validation = prepare_volatility_normalized_samples(
        _samples(
            np.array([0.02, -0.06, 0.03, 99.0]),
            np.array([0.01, 0.02, 0.03, 0.0]),
            start="2026-01-01",
        )
    )
    normalized_predictions = np.array([1.0, -2.0, 0.5])
    expected_raw_predictions = normalized_predictions * validation.sigma

    result = evaluate_vn_mse_experiment(
        train=train,
        validation=validation,
        normalized_predictions=normalized_predictions,
    )

    assert result["experiment_id"] == "E03-VN-MSE"
    assert result["evaluated_splits"] == ["validation"]
    assert result["lstm_validation"]["n"] == 3
    assert result["lstm_validation"]["mae"] == pytest.approx(
        np.mean(np.abs(validation.raw.targets - expected_raw_predictions))
    )
    assert result["raw_return_prediction_diagnostics"]["prediction_std"] == pytest.approx(
        np.std(expected_raw_predictions, ddof=0)
    )
    assert result["normalized_space_diagnostics"] == normalized_space_diagnostics(
        validation.normalized.targets,
        normalized_predictions,
    )
    assert result["data"]["validation_invalid_sigma_exclusions"] == 1
    assert result["comparability_to_e03_mse"][
        "same_validation_sample_identity_by_construction"
    ] is False
    assert "1 E03-MSE validation samples" in result["comparability_to_e03_mse"][
        "difference_reason"
    ]
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_vn_mse_preserves_every_e03_mse_setting_except_target_representation() -> None:
    baseline = resolve_mse_configuration(project_root=ROOT)
    resolved = resolve_vn_mse_configuration(project_root=ROOT)

    assert resolved.feature == baseline.feature
    assert resolved.model == baseline.model
    assert resolved.base_experiment == baseline.ablation_experiment
    assert _changed_leaf_paths(baseline.training, resolved.training) == [
        "loss.target_scale"
    ]
    assert resolved.training["loss"] == {
        "type": "torch.nn.MSELoss",
        "reduction": "mean",
        "target_scale": NORMALIZED_TARGET_SCALE,
    }
    baseline_without_scale = copy.deepcopy(baseline.training)
    resolved_without_scale = copy.deepcopy(resolved.training)
    baseline_without_scale["loss"].pop("target_scale")
    resolved_without_scale["loss"].pop("target_scale")
    assert resolved_without_scale == baseline_without_scale


def test_vn_mse_cli_has_a_distinct_run_identity() -> None:
    args = build_parser().parse_args(["run-lstm-volatility-normalized-mse"])

    assert args.command == "run-lstm-volatility-normalized-mse"
