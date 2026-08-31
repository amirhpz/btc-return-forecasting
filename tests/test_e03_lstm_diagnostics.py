from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_diagnostics import (
    OVERFIT_MAX_EPOCHS,
    OVERFIT_SAMPLE_COUNT,
    build_distribution_report,
    diagnostic_statistics,
    select_overfit_subset,
    validate_source_manifest,
)


def _samples(count: int, *, start: str, target_offset: float = 0.0) -> SequenceSamples:
    targets = np.linspace(-0.2, 0.2, count, dtype=np.float64) + target_offset
    return SequenceSamples(
        features=np.arange(count * 24 * 10, dtype=np.float32).reshape(count, 24, 10),
        targets=targets,
        decision_times=pd.date_range(start, periods=count, freq="h", tz="UTC"),
        candidate_count=count,
        excluded_lookback_count=0,
    )


def test_diagnostic_statistics_report_requested_distributions_and_residuals() -> None:
    targets = np.array([-2.0, -1.0, 1.0, 2.0])
    predictions = np.array([-1.0, 0.0, 2.0, 1.0])

    result = diagnostic_statistics(targets, predictions)

    assert result["model_metrics"]["n"] == 4
    assert result["target_distribution"] == {
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets, ddof=0)),
        "min": -2.0,
        "max": 2.0,
        "positive_ratio": 0.5,
    }
    assert result["prediction_distribution"] == {
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions, ddof=0)),
        "min": -1.0,
        "max": 2.0,
        "positive_ratio": 0.5,
    }
    residual = targets - predictions
    assert result["residual"]["definition"] == "target_minus_prediction"
    assert result["residual"]["mean"] == float(np.mean(residual))
    assert result["residual"]["std"] == float(np.std(residual, ddof=0))
    assert result["prediction_std_over_target_std"] == pytest.approx(
        np.std(predictions, ddof=0) / np.std(targets, ddof=0)
    )
    assert result["mean_abs_prediction_over_mean_abs_target"] == pytest.approx(
        np.mean(np.abs(predictions)) / np.mean(np.abs(targets))
    )


def test_constant_predictions_are_serializable_as_null_correlations() -> None:
    result = diagnostic_statistics(
        np.array([-1.0, 0.0, 1.0]),
        np.zeros(3),
    )

    assert result["model_metrics"]["pearson_ic"] is None
    assert result["model_metrics"]["spearman_rank_ic"] is None


def test_distribution_report_uses_train_and_validation_only_with_same_row_zero() -> None:
    train = _samples(5, start="2025-01-01")
    validation = _samples(4, start="2026-01-01", target_offset=0.01)
    train_predictions = train.targets * 0.8
    validation_predictions = validation.targets * 0.8

    result = build_distribution_report(
        train=train,
        train_predictions=train_predictions,
        validation=validation,
        validation_predictions=validation_predictions,
        source_run_id="E03_fixture",
        device="cuda",
        gpu_name="fixture_gpu",
    )

    assert result["evaluated_splits"] == ["train", "validation"]
    assert result["train"]["model_metrics"]["n"] == 5
    assert result["validation"]["model_metrics"]["n"] == 4
    assert result["validation"]["zero_return_same_rows"]["n"] == 4
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_overfit_subset_is_exactly_first_512_eligible_train_samples() -> None:
    train = _samples(600, start="2020-01-01")

    subset = select_overfit_subset(train)

    assert OVERFIT_SAMPLE_COUNT == 512
    assert OVERFIT_MAX_EPOCHS == 300
    assert len(subset.targets) == 512
    assert subset.features.shape == (512, 24, 10)
    assert np.array_equal(subset.targets, train.targets[:512])
    assert subset.decision_times.equals(train.decision_times[:512])
    with pytest.raises(ValueError, match="exactly 512"):
        select_overfit_subset(train, sample_count=511)


def test_source_manifest_rejects_any_test_access() -> None:
    valid = {
        "experiment_id": "E03",
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
    }
    validate_source_manifest(valid)

    with pytest.raises(ValueError, match="test set"):
        validate_source_manifest({**valid, "test_set": "EVALUATED"})
    with pytest.raises(ValueError, match="validation-only"):
        validate_source_manifest({**valid, "evaluated_splits": ["validation", "test"]})


def test_diagnostic_cli_commands_require_an_explicit_source_run() -> None:
    parser = build_parser()
    source = Path("outputs/runs/E03_fixture")

    distribution = parser.parse_args(
        ["diagnose-lstm-distribution", "--source-run", str(source)]
    )
    overfit = parser.parse_args(["diagnose-lstm-overfit", "--source-run", str(source)])

    assert distribution.source_run == source
    assert overfit.source_run == source


def test_diagnostic_helpers_do_not_modify_frozen_e03_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    config_paths = (
        root / "configs" / "experiments" / "e03.yaml",
        root / "configs" / "models" / "lstm.yaml",
        root / "configs" / "training.yaml",
    )
    before = {path: path.read_bytes() for path in config_paths}

    train = _samples(512, start="2020-01-01")
    select_overfit_subset(train)
    diagnostic_statistics(train.targets, train.targets * 0.5)

    assert {path: path.read_bytes() for path in config_paths} == before
