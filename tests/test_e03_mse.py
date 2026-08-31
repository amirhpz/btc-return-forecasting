from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from btc_forecasting.cli import build_parser
from btc_forecasting.common.config import load_yaml
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.training.lstm import (
    SequenceSamples,
    build_training_components,
    fit_lstm,
)
from btc_forecasting.training.lstm_mse import (
    EXPECTED_CHANGED_PATHS,
    _changed_leaf_paths,
    evaluate_mse_ablation,
    resolve_mse_configuration,
    validation_magnitude_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]


def _samples(count: int, *, start: str) -> SequenceSamples:
    generator = np.random.default_rng(42)
    return SequenceSamples(
        features=generator.normal(size=(count, 24, 10)).astype(np.float32),
        targets=generator.normal(scale=0.01, size=count).astype(np.float64),
        decision_times=pd.date_range(start, periods=count, freq="h", tz="UTC"),
        candidate_count=count,
        excluded_lookback_count=0,
    )


def test_mse_ablation_changes_only_loss_and_matching_monitor() -> None:
    original_paths = (
        ROOT / "configs" / "experiments" / "e03.yaml",
        ROOT / "configs" / "models" / "lstm.yaml",
        ROOT / "configs" / "training.yaml",
    )
    before = {path: path.read_bytes() for path in original_paths}
    base_training = load_yaml(ROOT / "configs" / "training.yaml")["training"]

    resolved = resolve_mse_configuration(project_root=ROOT)

    assert resolved.ablation_experiment["id"] == "E03-MSE"
    assert resolved.ablation_experiment["base_experiment"] == "E03"
    assert tuple(_changed_leaf_paths(base_training, resolved.training)) == EXPECTED_CHANGED_PATHS
    assert resolved.training["loss"] == {
        "type": "torch.nn.MSELoss",
        "reduction": "mean",
        "target_scale": "unscaled_one_hour_log_return",
    }
    assert resolved.training["early_stopping"]["monitor"] == "validation_mse_loss"
    assert {path: path.read_bytes() for path in original_paths} == before


def test_mse_components_preserve_optimizer_and_scheduler_semantics() -> None:
    resolved = resolve_mse_configuration(project_root=ROOT)
    model = LSTMRegressor(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        configured_dropout=0.20,
    )

    optimizer, loss, scheduler = build_training_components(
        model,
        training_config=resolved.training,
    )

    assert isinstance(loss, torch.nn.MSELoss)
    assert loss.reduction == "mean"
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["amsgrad"] is False
    assert scheduler.T_max == 30
    assert scheduler.eta_min == 0.0


def test_synthetic_fit_records_mse_loss_without_changing_model_shape() -> None:
    resolved = resolve_mse_configuration(project_root=ROOT)
    training = copy.deepcopy(resolved.training)
    training["max_epochs"] = 1
    model = LSTMRegressor(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        configured_dropout=0.20,
    )
    train = _samples(8, start="2025-01-01")
    validation = _samples(4, start="2026-01-01")

    outcome = fit_lstm(
        model,
        train=train,
        validation=validation,
        training_config=training,
        seed=42,
        device=torch.device("cpu"),
    )

    assert outcome.epochs_trained == 1
    assert "train_mse_loss" in outcome.history[0]
    assert "validation_mse_loss" in outcome.history[0]
    assert "train_huber_loss" not in outcome.history[0]
    assert outcome.model(torch.zeros((3, 24, 10))).shape == (3,)


def test_validation_magnitude_diagnostics_are_exact() -> None:
    targets = np.array([-2.0, -1.0, 1.0, 2.0])
    predictions = np.array([-1.0, 0.0, 1.0, 1.0])

    result = validation_magnitude_diagnostics(targets, predictions)

    assert result["target_mean"] == float(np.mean(targets))
    assert result["target_std"] == float(np.std(targets, ddof=0))
    assert result["prediction_mean"] == float(np.mean(predictions))
    assert result["prediction_std"] == float(np.std(predictions, ddof=0))
    assert result["prediction_std_over_target_std"] == pytest.approx(
        np.std(predictions, ddof=0) / np.std(targets, ddof=0)
    )
    assert result["mean_abs_prediction_over_mean_abs_target"] == pytest.approx(
        np.mean(np.abs(predictions)) / np.mean(np.abs(targets))
    )
    assert result["prediction_positive_ratio"] == 0.5
    assert result["target_positive_ratio"] == 0.5


def test_mse_evaluation_uses_validation_and_same_row_zero_without_test() -> None:
    train = _samples(5, start="2025-01-01")
    validation = _samples(4, start="2026-01-01")
    predictions = validation.targets * 0.5

    result = evaluate_mse_ablation(
        train=train,
        validation=validation,
        predictions=predictions,
    )

    assert result["experiment_id"] == "E03-MSE"
    assert result["evaluated_splits"] == ["validation"]
    assert result["lstm_validation"]["n"] == 4
    assert result["zero_return_same_validation_rows"]["n"] == 4
    assert result["validation_magnitude_diagnostics"]["target_std"] == pytest.approx(
        np.std(validation.targets, ddof=0)
    )
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_mse_cli_has_a_separate_run_identity() -> None:
    args = build_parser().parse_args(["run-lstm-mse-ablation"])

    assert args.command == "run-lstm-mse-ablation"
