from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_diagnostics import (
    OVERFIT_MAX_EPOCHS,
    select_overfit_subset,
)
from btc_forecasting.training.lstm_vn_learning_diagnostic import (
    build_learning_diagnostic_report,
    build_overfit_summary,
    normalized_target_metrics,
    restored_checkpoint_split_report,
    validate_vn_mse_source_manifest,
)
from btc_forecasting.training.lstm_vn_mse import (
    prepare_volatility_normalized_samples,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
)


def _samples(
    targets: np.ndarray,
    sigma: np.ndarray,
    *,
    start: str,
) -> SequenceSamples:
    truth = np.asarray(targets, dtype=np.float64)
    volatility = np.asarray(sigma, dtype=np.float64)
    features = np.zeros((len(truth), 24, len(F0_FEATURE_NAMES)), dtype=np.float32)
    features[:, -1, VOLATILITY_FEATURE_INDEX] = volatility
    return SequenceSamples(
        features=features,
        targets=truth,
        decision_times=pd.date_range(start, periods=len(truth), freq="h", tz="UTC"),
        candidate_count=len(truth),
        excluded_lookback_count=0,
    )


def test_source_manifest_requires_validation_only_vn_mse_without_test_access() -> None:
    valid = {
        "experiment_id": "E03-VN-MSE",
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
        "controlled_difference": {"epsilon_added": False},
    }

    validate_vn_mse_source_manifest(valid)

    with pytest.raises(ValueError, match="E03-VN-MSE"):
        validate_vn_mse_source_manifest({**valid, "experiment_id": "E03-MSE"})
    with pytest.raises(ValueError, match="test set"):
        validate_vn_mse_source_manifest({**valid, "test_set": "EVALUATED"})
    with pytest.raises(ValueError, match="validation-only"):
        validate_vn_mse_source_manifest(
            {**valid, "evaluated_splits": ["validation", "test"]}
        )
    with pytest.raises(ValueError, match="no-epsilon"):
        validate_vn_mse_source_manifest(
            {**valid, "controlled_difference": {"epsilon_added": True}}
        )


def test_normalized_target_metrics_include_zero_baseline_and_skills() -> None:
    targets = np.array([-2.0, -1.0, 1.0, 2.0])
    predictions = np.array([-1.0, -0.5, 0.5, 1.0])

    result = normalized_target_metrics(targets, predictions)

    expected_mse = float(np.mean(np.square(targets - predictions)))
    expected_mae = float(np.mean(np.abs(targets - predictions)))
    zero_mse = float(np.mean(np.square(targets)))
    zero_mae = float(np.mean(np.abs(targets)))
    assert result["n"] == 4
    assert result["mse"] == pytest.approx(expected_mse)
    assert result["mae"] == pytest.approx(expected_mae)
    assert result["target_mean"] == pytest.approx(np.mean(targets))
    assert result["target_std"] == pytest.approx(np.std(targets, ddof=0))
    assert result["prediction_std_over_target_std"] == pytest.approx(0.5)
    assert result["zero_same_rows"] == pytest.approx(
        {"mse": zero_mse, "mae": zero_mae}
    )
    assert result["skill"]["mse"] == pytest.approx(1.0 - expected_mse / zero_mse)
    assert result["skill"]["mae"] == pytest.approx(1.0 - expected_mae / zero_mae)


def test_restored_report_evaluates_z_and_reconstructed_raw_returns() -> None:
    prepared = prepare_volatility_normalized_samples(
        _samples(
            np.array([0.02, -0.06, 0.03, 99.0]),
            np.array([0.01, 0.02, 0.03, 0.0]),
            start="2026-01-01",
        )
    )
    z_predictions = np.array([1.0, -2.0, 0.5])
    raw_predictions = z_predictions * prepared.sigma

    result = restored_checkpoint_split_report(prepared, z_predictions)

    assert result["normalized_target"]["n"] == 3
    assert result["raw_reconstructed_return"]["n"] == 3
    assert result["raw_reconstructed_return"]["mae"] == pytest.approx(
        np.mean(np.abs(prepared.raw.targets - raw_predictions))
    )
    assert result["invalid_sigma_exclusions"] == 1


def test_overfit_summary_uses_first_512_chronological_normalized_samples() -> None:
    count = 520
    train = _samples(
        np.linspace(-0.2, 0.2, count),
        np.full(count, 0.1),
        start="2020-01-01",
    )
    normalized = prepare_volatility_normalized_samples(train).normalized
    subset = select_overfit_subset(normalized)
    initial_predictions = np.zeros(512)
    final_predictions = subset.targets * 0.75

    result = build_overfit_summary(
        subset=subset,
        initial_predictions=initial_predictions,
        final_predictions=final_predictions,
        best_mse=0.01,
        epochs_executed=OVERFIT_MAX_EPOCHS,
    )

    assert result["sample_count"] == 512
    assert result["first_decision_time"] == normalized.decision_times[0].isoformat()
    assert result["last_decision_time"] == normalized.decision_times[511].isoformat()
    assert result["initial_mse"] == pytest.approx(np.mean(np.square(subset.targets)))
    assert result["final_mae"] == pytest.approx(
        np.mean(np.abs(subset.targets - final_predictions))
    )
    assert result["final_prediction_std"] == pytest.approx(
        np.std(final_predictions, ddof=0)
    )
    assert result["epochs_executed"] == 300
    assert result["validation_set"] == "NOT USED"
    assert result["test_set"] == "NOT EVALUATED"


def test_learning_report_contains_train_validation_and_no_test_evaluation() -> None:
    split_report = {
        "normalized_target": {"n": 4},
        "raw_reconstructed_return": {"n": 4},
    }
    overfit_report = {"sample_count": 512, "validation_set": "NOT USED"}

    result = build_learning_diagnostic_report(
        source_run_id="E03-VN-MSE_fixture",
        train_report=split_report,
        validation_report=split_report,
        overfit_report=overfit_report,
        device="cuda",
        gpu_name="fixture_gpu",
        seed=42,
    )

    assert result["restored_checkpoint"]["evaluated_splits"] == [
        "train",
        "validation",
    ]
    assert result["normalized_target_overfit_sanity"]["validation_set"] == "NOT USED"
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_vn_learning_cli_requires_one_source_run() -> None:
    source = Path("outputs/runs/E03-VN-MSE_fixture")
    args = build_parser().parse_args(
        ["diagnose-lstm-vn-learning", "--source-run", str(source)]
    )

    assert args.command == "diagnose-lstm-vn-learning"
    assert args.source_run == source
