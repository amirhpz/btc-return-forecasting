from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm_classifier import LSTMClassifier
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_classification_walkforward import (
    CLASS_IDS,
    CLASS_NAMES,
    LABEL_METHODS,
    ClassificationData,
    ClassificationTrainingOutcome,
    FixedClassificationOutcome,
    LabelCalibration,
    aggregate_classification_metrics,
    baseline_predictions,
    build_classification_components,
    build_classification_temporal_design,
    calibrate_and_apply_labels,
    classification_stability_gate,
    resolve_classification_configuration,
    select_classification_label,
    train_label_fold_seed,
    validate_e05_label_audit,
)
from btc_forecasting.training.lstm_horizon_walkforward import (
    EXPECTED_COMMON_COUNT,
    EXPECTED_COMMON_IDENTITY,
    PairedHorizonSamples,
)
from btc_forecasting.training.lstm_vn_multiseed import FROZEN_SEEDS
from btc_forecasting.training.lstm_vn_walkforward import OuterFold


def test_exact_labels_class_order_model_head_loss_and_seeds() -> None:
    model = LSTMClassifier(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        configured_dropout=0.20,
    )
    logits = model(torch.zeros(4, 24, 10))
    configuration = resolve_classification_configuration(Path.cwd())
    optimizer, loss, _ = build_classification_components(
        model,
        training_config=configuration["training"],  # type: ignore[arg-type]
    )

    assert LABEL_METHODS == ("L1_FIXED", "L2_ATR_ADAPTIVE")
    assert "L3_ATR_HYSTERESIS_3" not in LABEL_METHODS
    assert CLASS_IDS == (0, 1, 2)
    assert CLASS_NAMES == ("DOWN", "NEUTRAL", "UP")
    assert logits.shape == (4, 3)
    assert model.classification_head.out_features == 3
    assert model.effective_lstm_dropout == 0.0
    assert isinstance(loss, torch.nn.CrossEntropyLoss)
    assert loss.weight is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert configuration["training"]["early_stopping"]["monitor"] == "validation_cross_entropy_loss"  # type: ignore[index]
    assert FROZEN_SEEDS == (42, 137, 271, 811, 2027)


def test_calibration_uses_train_only_and_maps_down_neutral_up_to_0_1_2() -> None:
    calibration_y = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    calibration_atr = np.ones(7)
    applied_y = np.array([-4.0, 0.0, 4.0])
    applied_atr = np.ones(3)
    for method, parameter in (("L1_FIXED", "tau"), ("L2_ATR_ADAPTIVE", "k")):
        first = calibrate_and_apply_labels(
            method=method,
            calibration_returns=calibration_y,
            calibration_atr=calibration_atr,
            applied_returns=applied_y,
            applied_atr=applied_atr,
        )
        changed_evaluation = calibrate_and_apply_labels(
            method=method,
            calibration_returns=calibration_y,
            calibration_atr=calibration_atr,
            applied_returns=applied_y * 1_000_000.0,
            applied_atr=applied_atr,
        )
        assert first.parameter_name == parameter
        assert first.parameter_value == changed_evaluation.parameter_value
        assert tuple(first.applied_labels) == (0, 1, 2)
        assert set(np.unique(first.train_labels)).issubset(CLASS_IDS)


def test_six_paired_blocks_four_folds_and_one_hour_boundary_purge() -> None:
    times = pd.date_range("2020-01-01", periods=121, freq="h", tz="UTC")
    blocks, folds = build_classification_temporal_design(times)
    assert len(blocks) == 6
    assert len(folds) == 4
    assert np.array_equal(np.concatenate([block.positions for block in blocks]), np.arange(121))
    for fold, evaluation_block in zip(folds, blocks[2:], strict=True):
        assert np.array_equal(fold.outer_evaluation_positions, evaluation_block.positions)
        inner_start = times[fold.inner_validation_positions[0]]
        outer_start = times[fold.outer_evaluation_positions[0]]
        assert times[fold.inner_train_positions[-1]] + np.timedelta64(1, "h") < inner_start
        assert times[fold.outer_pool_positions[-1]] + np.timedelta64(1, "h") < outer_start


def _classification_data() -> ClassificationData:
    count = 30
    times = pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC")
    returns = np.linspace(-0.03, 0.03, count)
    return ClassificationData(
        samples=PairedHorizonSamples(
            features=np.zeros((count, 24, len(F0_FEATURE_NAMES)), dtype=np.float32),
            decision_times=times,
            sigma=np.full(count, 0.01),
            raw_targets={1: returns},
            normalized_targets={1: returns / 0.01},
        ),
        atr_pct_14=np.full(count, 0.02),
    )


def test_inner_outer_calibration_scalers_and_epoch_selection_are_scope_isolated(monkeypatch) -> None:
    data = _classification_data()
    fold = OuterFold(
        number=1,
        outer_pool_positions=np.arange(20),
        inner_train_positions=np.arange(12),
        inner_validation_positions=np.arange(12, 16),
        outer_evaluation_positions=np.arange(20, 25),
        purged_outer_boundary_count=1,
        purged_inner_boundary_count=1,
    )
    calibration_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    scaler_fit_times: list[pd.DatetimeIndex] = []
    events: list[str] = []

    def fake_calibration(**kwargs):
        calibration_inputs.append(
            (kwargs["calibration_returns"].copy(), kwargs["applied_returns"].copy())
        )
        train_count = len(kwargs["calibration_returns"])
        applied_count = len(kwargs["applied_returns"])
        return LabelCalibration(
            kwargs["method"],
            "tau",
            float(train_count),
            np.arange(train_count) % 3,
            np.arange(applied_count) % 3,
        )

    def fake_fit_scaler(samples: SequenceSamples):
        scaler_fit_times.append(samples.decision_times)
        events.append("fit_scaler")
        return object(), samples

    def fake_transform(samples: SequenceSamples, scaler: object):
        if samples.decision_times.equals(
            data.samples.decision_times[fold.outer_evaluation_positions]
        ):
            events.append("transform_outer_evaluation")
        return samples

    def fake_inner(model, *, train, validation, training_config, seed, device):
        assert validation.decision_times.equals(
            data.samples.decision_times[fold.inner_validation_positions]
        )
        events.append("inner_selection")
        return ClassificationTrainingOutcome(
            model=model,
            history=[{"epoch": 1}],
            best_epoch=2,
            epochs_trained=3,
            best_validation_loss=0.5,
            duration_seconds=1.0,
        )

    def fake_refit(model, *, train, training_config, seed, device, epochs):
        assert epochs == 2
        assert train.decision_times.equals(data.samples.decision_times[fold.outer_pool_positions])
        events.append("outer_refit")
        return FixedClassificationOutcome(model, [{"epoch": 1}, {"epoch": 2}], 2.0)

    probabilities = np.tile(np.array([[0.2, 0.3, 0.5]]), (5, 1))
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.calibrate_and_apply_labels",
        fake_calibration,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.configure_determinism",
        lambda seed: None,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward._new_model",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.fit_scope_scaler",
        fake_fit_scaler,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.transform_scope",
        fake_transform,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.fit_classification_lstm",
        fake_inner,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.fit_classification_fixed_epochs",
        fake_refit,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_classification_walkforward.predict_class_probabilities",
        lambda model, features, batch_size, device: probabilities,
    )

    trained = train_label_fold_seed(
        method="L1_FIXED",
        fold=fold,
        seed=42,
        data=data,
        model_config={"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        training_config={"batch_size": 128, "max_epochs": 30},
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(
        calibration_inputs[0][0], data.samples.raw_targets[1][fold.inner_train_positions]
    )
    np.testing.assert_array_equal(
        calibration_inputs[0][1], data.samples.raw_targets[1][fold.inner_validation_positions]
    )
    np.testing.assert_array_equal(
        calibration_inputs[1][0], data.samples.raw_targets[1][fold.outer_pool_positions]
    )
    np.testing.assert_array_equal(
        calibration_inputs[1][1], data.samples.raw_targets[1][fold.outer_evaluation_positions]
    )
    assert scaler_fit_times[0].equals(data.samples.decision_times[fold.inner_train_positions])
    assert scaler_fit_times[1].equals(data.samples.decision_times[fold.outer_pool_positions])
    assert events.index("outer_refit") < events.index("transform_outer_evaluation")
    assert trained.metrics["training"]["inner_best_epoch"] == 2  # type: ignore[index]


def test_baselines_are_constructed_from_outer_train_labels_only() -> None:
    train = np.array([0, 0, 0, 1, 2])
    first = baseline_predictions(train, 4)
    second = baseline_predictions(train, 4)
    assert first["train_majority_class"] == 0
    assert first["train_class_prior"] == [0.6, 0.2, 0.2]
    np.testing.assert_array_equal(first["train_majority_predictions"], np.zeros(4))
    np.testing.assert_array_equal(first["always_neutral_predictions"], np.ones(4))
    np.testing.assert_allclose(first["train_prior_probabilities"], second["train_prior_probabilities"])


def _aggregate(
    *,
    balanced: float,
    mcc: float,
    macro_f1: float,
    baseline_f1: float,
) -> dict[str, dict[str, float]]:
    result = {
        name: {
            "mean": 0.0,
            "median": 0.0,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
        for name in (
            "balanced_accuracy",
            "mcc",
            "macro_f1",
            "train_majority_macro_f1",
        )
    }
    result["balanced_accuracy"]["mean"] = balanced
    result["mcc"]["mean"] = mcc
    result["macro_f1"]["mean"] = macro_f1
    result["train_majority_macro_f1"]["mean"] = baseline_f1
    return result


def _label_result(stable: bool, folds: int, balanced: float, macro_f1: float, mcc: float):
    overall = _aggregate(
        balanced=balanced,
        mcc=mcc,
        macro_f1=macro_f1,
        baseline_f1=0.30,
    )
    return {
        "CLASSIFICATION_TEMPORAL_STABLE": stable,
        "classification_positive_fold_count": folds,
        "overall_20_evaluation_aggregate": overall,
    }


def test_classification_gate_and_label_winner_rules_are_exact() -> None:
    positive = _aggregate(balanced=0.34, mcc=0.01, macro_f1=0.32, baseline_f1=0.31)
    negative = _aggregate(balanced=0.34, mcc=0.0, macro_f1=0.32, baseline_f1=0.31)
    flags, stable = classification_stability_gate(
        [positive, positive, positive, negative],
        positive,
    )
    assert flags == [True, True, True, False]
    assert stable is True
    _, unstable = classification_stability_gate(
        [positive, positive, negative, negative],
        positive,
    )
    assert unstable is False

    none = {
        method: _label_result(False, 4, 0.50, 0.50, 0.50) for method in LABEL_METHODS
    }
    assert select_classification_label(none) is None
    ranked = {
        "L1_FIXED": _label_result(True, 3, 0.40, 0.40, 0.10),
        "L2_ATR_ADAPTIVE": _label_result(True, 4, 0.35, 0.35, 0.05),
    }
    assert select_classification_label(ranked) == "L2_ATR_ADAPTIVE"


def test_e05_gate_requires_closed_validation_and_test(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "data" / "trend_label_audit" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "audit_id": "E05-L-A",
                "frozen_problem": {
                    "common_anchor_count": EXPECTED_COMMON_COUNT,
                    "common_anchor_identity_sha256": EXPECTED_COMMON_IDENTITY,
                },
                "feasibility": {
                    "L1_FIXED": {"result": "FEASIBLE"},
                    "L2_ATR_ADAPTIVE": {"result": "FEASIBLE"},
                },
                "l3_semantic_degradation": {"SEMANTIC_DEGRADATION": True},
                "original_validation": "NOT READ OR USED",
                "test_set": "NOT READ OR USED",
            }
        ),
        encoding="utf-8",
    )
    audit = validate_e05_label_audit(tmp_path)
    assert audit["original_validation"] == "NOT READ OR USED"
    assert audit["test_set"] == "NOT READ OR USED"
