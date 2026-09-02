from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from btc_forecasting.common.config import load_yaml
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm_classifier import LSTMClassifier
from btc_forecasting.targets.horizon_audit import _sample_identity
from btc_forecasting.targets.trend_label_audit import (
    AUDIT_OUTPUT_RELATIVE_PATH as LABEL_AUDIT_RELATIVE_PATH,
    DOWN,
    NEUTRAL,
    UP,
    _atr_on_common_anchors,
    adaptive_three_class_labels,
    calibrate_adaptive_k,
    calibrate_fixed_tau,
    fixed_three_class_labels,
    temporal_behavior,
)
from btc_forecasting.training.lstm import (
    F0_CONFIG_RELATIVE_PATH,
    LOOKBACK_HOURS,
    LSTM_CONFIG_RELATIVE_PATH,
    TRAINING_CONFIG_RELATIVE_PATH,
    LSTMRunResult,
    SequenceSamples,
    _write_json,
    configure_determinism,
    require_official_cuda,
)
from btc_forecasting.training.lstm_horizon_walkforward import (
    EXPECTED_COMMON_COUNT,
    EXPECTED_COMMON_IDENTITY,
    PairedHorizonSamples,
    load_paired_horizon_samples,
)
from btc_forecasting.training.lstm_vn_multiseed import FROZEN_SEEDS
from btc_forecasting.training.lstm_vn_walkforward import (
    BLOCK_COUNT,
    FOLD_COUNT,
    INNER_VALIDATION_FRACTION,
    OuterFold,
    TemporalBlock,
    fit_scope_scaler,
    selected_refit_epoch_count,
    transform_scope,
)

EXPERIMENT_ID = "E06-C-WF4"
LABEL_METHODS = ("L1_FIXED", "L2_ATR_ADAPTIVE")
CLASS_NAMES = ("DOWN", "NEUTRAL", "UP")
CLASS_IDS = (0, 1, 2)
ONE_HOUR = timedelta(hours=1)
PRIMARY_METRIC_PATHS = {
    "accuracy": ("classification", "accuracy"),
    "balanced_accuracy": ("classification", "balanced_accuracy"),
    "macro_f1": ("classification", "macro_f1"),
    "mcc": ("classification", "mcc"),
    "macro_precision": ("classification", "macro_precision"),
    "macro_recall": ("classification", "macro_recall"),
    "log_loss": ("probability", "log_loss"),
    "multiclass_brier": ("probability", "multiclass_brier"),
    "train_majority_accuracy": ("baselines", "train_majority_accuracy"),
    "train_majority_balanced_accuracy": (
        "baselines",
        "train_majority_balanced_accuracy",
    ),
    "train_majority_macro_f1": ("baselines", "train_majority_macro_f1"),
    "train_majority_mcc": ("baselines", "train_majority_mcc"),
    "always_neutral_accuracy": ("baselines", "always_neutral_accuracy"),
    "always_neutral_balanced_accuracy": (
        "baselines",
        "always_neutral_balanced_accuracy",
    ),
    "always_neutral_macro_f1": ("baselines", "always_neutral_macro_f1"),
    "always_neutral_mcc": ("baselines", "always_neutral_mcc"),
    "train_prior_log_loss": ("baselines", "train_prior_log_loss"),
    "train_prior_multiclass_brier": (
        "baselines",
        "train_prior_multiclass_brier",
    ),
}


@dataclass(frozen=True)
class ClassificationData:
    samples: PairedHorizonSamples
    atr_pct_14: np.ndarray


@dataclass(frozen=True)
class LabelCalibration:
    method: str
    parameter_name: str
    parameter_value: float
    train_labels: np.ndarray
    applied_labels: np.ndarray


@dataclass(frozen=True)
class ClassificationTrainingOutcome:
    model: LSTMClassifier
    history: list[dict[str, float | int]]
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float
    duration_seconds: float


@dataclass(frozen=True)
class FixedClassificationOutcome:
    model: LSTMClassifier
    history: list[dict[str, float | int]]
    duration_seconds: float


@dataclass(frozen=True)
class LabelFoldSeedTraining:
    label_method: str
    fold: int
    seed: int
    inner_history: list[dict[str, float | int]]
    refit_history: list[dict[str, float | int]]
    probabilities: np.ndarray
    predicted_classes: np.ndarray
    true_classes: np.ndarray
    raw_future_returns: np.ndarray
    metrics: dict[str, object]


def validate_e05_label_audit(project_root: Path) -> dict[str, object]:
    path = project_root.resolve() / LABEL_AUDIT_RELATIVE_PATH
    with path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("audit_id") != "E05-L-A":
        raise ValueError("E06 requires the validated E05-L-A artifact")
    problem = audit.get("frozen_problem", {})
    if problem.get("common_anchor_count") != EXPECTED_COMMON_COUNT:
        raise ValueError("E05 COMMON anchor count does not match E06")
    if problem.get("common_anchor_identity_sha256") != EXPECTED_COMMON_IDENTITY:
        raise ValueError("E05 COMMON anchor identity does not match E06")
    feasibility = audit.get("feasibility", {})
    for method in LABEL_METHODS:
        if feasibility.get(method, {}).get("result") != "FEASIBLE":
            raise ValueError(f"E06 requires E05 feasibility for {method}")
    degradation = audit.get("l3_semantic_degradation", {})
    if degradation.get("SEMANTIC_DEGRADATION") is not True:
        raise ValueError("E06 expects the validated L3 semantic-degradation exclusion")
    if audit.get("original_validation") != "NOT READ OR USED":
        raise ValueError("E05 artifact did not preserve original Validation closure")
    if audit.get("test_set") != "NOT READ OR USED":
        raise ValueError("E05 artifact did not preserve TEST closure")
    return audit


def resolve_classification_configuration(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    feature = load_yaml(root / F0_CONFIG_RELATIVE_PATH)["feature_set"]
    base_model = load_yaml(root / LSTM_CONFIG_RELATIVE_PATH)["model"]
    base_training = load_yaml(root / TRAINING_CONFIG_RELATIVE_PATH)["training"]
    if tuple(feature["features"]) != F0_FEATURE_NAMES:
        raise ValueError("E06 requires the frozen F0 feature order")
    model = copy.deepcopy(base_model)
    model["output_size"] = 3
    model.pop("regression_head", None)
    model["classification_head"] = {
        "type": "linear",
        "in_features": "hidden_size",
        "out_features": 3,
        "activation": "none_raw_logits",
        "class_order": list(CLASS_NAMES),
    }
    training = copy.deepcopy(base_training)
    training["loss"] = {
        "type": "torch.nn.CrossEntropyLoss",
        "reduction": "mean",
        "class_weights": None,
        "target": "class_ids_0_down_1_neutral_2_up",
    }
    training["early_stopping"]["monitor"] = "validation_cross_entropy_loss"
    return {"feature_set": feature, "model": model, "training": training}


def load_classification_data(*, project_root: Path) -> ClassificationData:
    root = project_root.resolve()
    validate_e05_label_audit(root)
    samples = load_paired_horizon_samples(project_root=root)
    if len(samples.decision_times) != EXPECTED_COMMON_COUNT:
        raise ValueError("E06 requires exactly 53,013 COMMON TRAIN anchors")
    if _sample_identity(samples.decision_times) != EXPECTED_COMMON_IDENTITY:
        raise ValueError("E06 COMMON TRAIN identity mismatch")
    atr = _atr_on_common_anchors(
        project_root=root,
        decision_times=samples.decision_times,
    )
    return ClassificationData(samples=samples, atr_pct_14=atr)


def _purge_one_hour_targets(
    positions: np.ndarray,
    decision_times: pd.DatetimeIndex,
    boundary: pd.Timestamp,
) -> np.ndarray:
    return positions[decision_times[positions] + ONE_HOUR < boundary]


def build_classification_temporal_design(
    decision_times: pd.DatetimeIndex,
) -> tuple[list[TemporalBlock], list[OuterFold]]:
    if len(decision_times) < BLOCK_COUNT:
        raise ValueError("E06 requires six non-empty COMMON blocks")
    if decision_times.has_duplicates or not decision_times.is_monotonic_increasing:
        raise ValueError("E06 decision times must be strictly ordered and unique")
    blocks = [
        TemporalBlock(number=index, positions=positions)
        for index, positions in enumerate(
            np.array_split(np.arange(len(decision_times)), BLOCK_COUNT),
            1,
        )
    ]
    folds: list[OuterFold] = []
    for fold_number in range(1, FOLD_COUNT + 1):
        outer_evaluation = blocks[fold_number + 1].positions
        outer_start = decision_times[outer_evaluation[0]]
        unpurged_outer = np.concatenate(
            [block.positions for block in blocks[: fold_number + 1]]
        )
        outer_pool = _purge_one_hour_targets(
            unpurged_outer,
            decision_times,
            outer_start,
        )
        inner_count = max(1, math.floor(len(outer_pool) * INNER_VALIDATION_FRACTION))
        inner_validation = outer_pool[-inner_count:]
        unpurged_inner = outer_pool[:-inner_count]
        inner_start = decision_times[inner_validation[0]]
        inner_train = _purge_one_hour_targets(
            unpurged_inner,
            decision_times,
            inner_start,
        )
        folds.append(
            OuterFold(
                number=fold_number,
                outer_pool_positions=outer_pool,
                inner_train_positions=inner_train,
                inner_validation_positions=inner_validation,
                outer_evaluation_positions=outer_evaluation,
                purged_outer_boundary_count=len(unpurged_outer) - len(outer_pool),
                purged_inner_boundary_count=len(unpurged_inner) - len(inner_train),
            )
        )
    return blocks, folds


def calibrate_and_apply_labels(
    *,
    method: str,
    calibration_returns: np.ndarray,
    calibration_atr: np.ndarray,
    applied_returns: np.ndarray,
    applied_atr: np.ndarray,
) -> LabelCalibration:
    if method == "L1_FIXED":
        tau = calibrate_fixed_tau(calibration_returns)
        return LabelCalibration(
            method=method,
            parameter_name="tau",
            parameter_value=tau,
            train_labels=fixed_three_class_labels(calibration_returns, tau=tau) + 1,
            applied_labels=fixed_three_class_labels(applied_returns, tau=tau) + 1,
        )
    if method == "L2_ATR_ADAPTIVE":
        k = calibrate_adaptive_k(calibration_returns, calibration_atr)
        train_labels, _ = adaptive_three_class_labels(
            calibration_returns,
            calibration_atr,
            k=k,
        )
        applied_labels, _ = adaptive_three_class_labels(
            applied_returns,
            applied_atr,
            k=k,
        )
        return LabelCalibration(method, "k", k, train_labels + 1, applied_labels + 1)
    raise ValueError(f"E06 compares only L1 and L2, not {method!r}")


def _sequence(
    data: ClassificationData,
    positions: np.ndarray,
    labels: np.ndarray,
) -> SequenceSamples:
    class_ids = np.asarray(labels, dtype=np.int64)
    if set(np.unique(class_ids)) - set(CLASS_IDS):
        raise ValueError("Classification targets must be class IDs 0, 1, 2")
    return SequenceSamples(
        features=data.samples.features[positions],
        targets=class_ids,
        decision_times=data.samples.decision_times[positions],
        candidate_count=len(positions),
        excluded_lookback_count=0,
    )


def _classification_loader(
    samples: SequenceSamples,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(samples.features),
        torch.from_numpy(samples.targets.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def build_classification_components(
    model: LSTMClassifier,
    *,
    training_config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, nn.CrossEntropyLoss, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer_config = training_config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    loss_config = training_config["loss"]
    if loss_config != {
        "type": "torch.nn.CrossEntropyLoss",
        "reduction": "mean",
        "class_weights": None,
        "target": "class_ids_0_down_1_neutral_2_up",
    }:
        raise ValueError("E06 requires unweighted mean CrossEntropyLoss")
    loss = nn.CrossEntropyLoss(weight=None, reduction="mean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["max_epochs"]),
        eta_min=float(training_config["scheduler"]["eta_min"]),
    )
    return optimizer, loss, scheduler


def _new_model(model_config: dict[str, Any]) -> LSTMClassifier:
    return LSTMClassifier(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        configured_dropout=float(model_config["dropout"]),
        class_count=3,
    )


def fit_classification_lstm(
    model: LSTMClassifier,
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> ClassificationTrainingOutcome:
    if training_config["early_stopping"]["monitor"] != "validation_cross_entropy_loss":
        raise ValueError("E06 early stopping must monitor inner validation cross-entropy")
    batch_size = int(training_config["batch_size"])
    train_loader = _classification_loader(
        train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _classification_loader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    model.to(device)
    optimizer, loss_function, scheduler = build_classification_components(
        model,
        training_config=training_config,
    )
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping"]["patience"])
    max_norm = float(training_config["gradient_clipping"]["max_norm"])
    norm_type = float(training_config["gradient_clipping"]["norm_type"])
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            optimizer.step()
            train_sum += float(loss.detach().item()) * len(labels)
            train_count += len(labels)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                loss = loss_function(model(features), labels)
                validation_sum += float(loss.item()) * len(labels)
                validation_count += len(labels)
        train_loss = train_sum / train_count
        validation_loss = validation_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_cross_entropy_loss": train_loss,
                "validation_cross_entropy_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        scheduler.step()
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("E06 inner selection completed without a best checkpoint")
    model.load_state_dict(best_state)
    return ClassificationTrainingOutcome(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_trained=len(history),
        best_validation_loss=best_loss,
        duration_seconds=time.perf_counter() - started,
    )


def fit_classification_fixed_epochs(
    model: LSTMClassifier,
    *,
    train: SequenceSamples,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
) -> FixedClassificationOutcome:
    epoch_count = selected_refit_epoch_count(
        epochs,
        max_epochs=int(training_config["max_epochs"]),
    )
    loader = _classification_loader(
        train,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model.to(device)
    optimizer, loss_function, scheduler = build_classification_components(
        model,
        training_config=training_config,
    )
    max_norm = float(training_config["gradient_clipping"]["max_norm"])
    norm_type = float(training_config["gradient_clipping"]["norm_type"])
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, epoch_count + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(labels)
            count += len(labels)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_cross_entropy_loss": loss_sum / count,
            }
        )
        scheduler.step()
    return FixedClassificationOutcome(model, history, time.perf_counter() - started)


def predict_class_probabilities(
    model: LSTMClassifier,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if result.shape != (len(features), 3):
        raise ValueError("E06 model must return exactly three class probabilities")
    return result


def _hard_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(CLASS_IDS),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def multiclass_brier_score(true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(3, dtype=np.float64)[np.asarray(true, dtype=np.int64)]
    return float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))


def baseline_predictions(
    train_labels: np.ndarray,
    evaluation_count: int,
) -> dict[str, np.ndarray | int | list[float]]:
    train = np.asarray(train_labels, dtype=np.int64)
    if set(np.unique(train)) - set(CLASS_IDS) or evaluation_count <= 0:
        raise ValueError("Baseline construction requires train class IDs and evaluation rows")
    counts = np.bincount(train, minlength=3)
    majority = int(np.argmax(counts))
    prior = counts.astype(np.float64) / len(train)
    return {
        "train_majority_class": majority,
        "train_class_prior": prior.tolist(),
        "train_majority_predictions": np.full(evaluation_count, majority, dtype=np.int64),
        "always_neutral_predictions": np.full(evaluation_count, 1, dtype=np.int64),
        "train_prior_probabilities": np.tile(prior, (evaluation_count, 1)),
    }


def evaluate_classification(
    *,
    true: np.ndarray,
    probabilities: np.ndarray,
    train_labels: np.ndarray,
    decision_times: pd.DatetimeIndex,
) -> dict[str, object]:
    true_ids = np.asarray(true, dtype=np.int64)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    primary = _hard_metrics(true_ids, predicted)
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        true_ids,
        predicted,
        labels=list(CLASS_IDS),
        average=None,
        zero_division=0,
    )
    baselines = baseline_predictions(train_labels, len(true_ids))
    majority_metrics = _hard_metrics(
        true_ids,
        baselines["train_majority_predictions"],  # type: ignore[arg-type]
    )
    neutral_metrics = _hard_metrics(
        true_ids,
        baselines["always_neutral_predictions"],  # type: ignore[arg-type]
    )
    prior_probabilities = baselines["train_prior_probabilities"]
    prior_log_loss = float(log_loss(true_ids, prior_probabilities, labels=list(CLASS_IDS)))  # type: ignore[arg-type]
    prior_brier = multiclass_brier_score(true_ids, prior_probabilities)  # type: ignore[arg-type]
    activity = temporal_behavior(predicted - 1, decision_times)
    return {
        "classification": {"n": len(true_ids), **primary},
        "per_class": {
            name: {
                "precision": float(per_precision[class_id]),
                "recall": float(per_recall[class_id]),
                "f1": float(per_f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            true_ids,
            predicted,
            labels=list(CLASS_IDS),
        ).tolist(),
        "predicted_class_distribution": {
            name: float(np.mean(predicted == class_id))
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "true_class_distribution": {
            name: float(np.mean(true_ids == class_id))
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "probability": {
            "log_loss": float(log_loss(true_ids, probabilities, labels=list(CLASS_IDS))),
            "multiclass_brier": multiclass_brier_score(true_ids, probabilities),
            "post_hoc_calibration": False,
        },
        "baselines": {
            "train_majority_class": baselines["train_majority_class"],
            "train_class_prior": baselines["train_class_prior"],
            **{f"train_majority_{key}": value for key, value in majority_metrics.items()},
            **{f"always_neutral_{key}": value for key, value in neutral_metrics.items()},
            "train_prior_log_loss": prior_log_loss,
            "train_prior_multiclass_brier": prior_brier,
            "fit_scope": "outer TRAIN labels only",
        },
        "prediction_activity": {
            "down_share": float(np.mean(predicted == 0)),
            "neutral_share": float(np.mean(predicted == 1)),
            "up_share": float(np.mean(predicted == 2)),
            "transition_count": activity["total_label_transitions"],
            "transitions_per_1000_hours": activity[
                "transitions_per_1000_valid_hours"
            ],
        },
    }


def train_label_fold_seed(
    *,
    method: str,
    fold: OuterFold,
    seed: int,
    data: ClassificationData,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> LabelFoldSeedTraining:
    returns = data.samples.raw_targets[1]
    inner_calibration = calibrate_and_apply_labels(
        method=method,
        calibration_returns=returns[fold.inner_train_positions],
        calibration_atr=data.atr_pct_14[fold.inner_train_positions],
        applied_returns=returns[fold.inner_validation_positions],
        applied_atr=data.atr_pct_14[fold.inner_validation_positions],
    )
    inner_train = _sequence(
        data,
        fold.inner_train_positions,
        inner_calibration.train_labels,
    )
    inner_validation = _sequence(
        data,
        fold.inner_validation_positions,
        inner_calibration.applied_labels,
    )
    configure_determinism(seed)
    inner_scaler, scaled_inner_train = fit_scope_scaler(inner_train)
    scaled_inner_validation = transform_scope(inner_validation, inner_scaler)
    inner_outcome = fit_classification_lstm(
        _new_model(model_config),
        train=scaled_inner_train,
        validation=scaled_inner_validation,
        training_config=training_config,
        seed=seed,
        device=device,
    )

    outer_calibration = calibrate_and_apply_labels(
        method=method,
        calibration_returns=returns[fold.outer_pool_positions],
        calibration_atr=data.atr_pct_14[fold.outer_pool_positions],
        applied_returns=returns[fold.outer_evaluation_positions],
        applied_atr=data.atr_pct_14[fold.outer_evaluation_positions],
    )
    outer_train = _sequence(
        data,
        fold.outer_pool_positions,
        outer_calibration.train_labels,
    )
    outer_evaluation = _sequence(
        data,
        fold.outer_evaluation_positions,
        outer_calibration.applied_labels,
    )
    configure_determinism(seed)
    outer_scaler, scaled_outer_train = fit_scope_scaler(outer_train)
    refit_outcome = fit_classification_fixed_epochs(
        _new_model(model_config),
        train=scaled_outer_train,
        training_config=training_config,
        seed=seed,
        device=device,
        epochs=inner_outcome.best_epoch,
    )
    scaled_outer_evaluation = transform_scope(outer_evaluation, outer_scaler)
    probabilities = predict_class_probabilities(
        refit_outcome.model,
        scaled_outer_evaluation.features,
        batch_size=int(training_config["batch_size"]),
        device=device,
    )
    evaluation = evaluate_classification(
        true=outer_calibration.applied_labels,
        probabilities=probabilities,
        train_labels=outer_calibration.train_labels,
        decision_times=outer_evaluation.decision_times,
    )
    metrics: dict[str, object] = {
        "label_method": method,
        "fold": fold.number,
        "seed": seed,
        "train_sample_count": len(fold.outer_pool_positions),
        "validation_sample_count": len(fold.outer_evaluation_positions),
        "train_sample_identity_sha256": _sample_identity(
            data.samples.decision_times[fold.outer_pool_positions]
        ),
        "validation_sample_identity_sha256": _sample_identity(
            data.samples.decision_times[fold.outer_evaluation_positions]
        ),
        **evaluation,
        "calibration": {
            "inner": {
                "parameter": inner_calibration.parameter_name,
                "value": inner_calibration.parameter_value,
                "fit_scope": "inner TRAIN only",
            },
            "outer": {
                "parameter": outer_calibration.parameter_name,
                "value": outer_calibration.parameter_value,
                "fit_scope": "complete leakage-purged outer TRAIN only",
            },
        },
        "training": {
            "inner_best_epoch": inner_outcome.best_epoch,
            "inner_epochs_trained": inner_outcome.epochs_trained,
            "best_inner_validation_cross_entropy": inner_outcome.best_validation_loss,
            "refit_epoch_count": len(refit_outcome.history),
            "inner_duration_seconds": inner_outcome.duration_seconds,
            "refit_duration_seconds": refit_outcome.duration_seconds,
        },
        "original_validation": "NOT READ OR USED",
        "test_set": "NOT READ OR USED",
    }
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    return LabelFoldSeedTraining(
        method,
        fold.number,
        seed,
        inner_outcome.history,
        refit_outcome.history,
        probabilities,
        predicted,
        outer_calibration.applied_labels.astype(np.int64),
        returns[fold.outer_evaluation_positions],
        metrics,
    )


def _metric(result: dict[str, object], path: tuple[str, str]) -> float:
    group = result[path[0]]
    if not isinstance(group, dict):
        raise TypeError(f"Metric group is not a mapping: {path[0]}")
    value = float(group[path[1]])
    if not np.isfinite(value):
        raise ValueError(f"Metric must be finite: {path}")
    return value


def aggregate_classification_metrics(
    evaluations: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    if not evaluations:
        raise ValueError("Classification aggregation requires evaluations")
    result: dict[str, dict[str, float]] = {}
    for name, path in PRIMARY_METRIC_PATHS.items():
        values = np.asarray([_metric(item, path) for item in evaluations])
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return result


def classification_stability_gate(
    fold_aggregates: list[dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> tuple[list[bool], bool]:
    if len(fold_aggregates) != 4:
        raise ValueError("Classification gate requires exactly four outer folds")
    positive = [
        fold["balanced_accuracy"]["mean"] > 1.0 / 3.0
        and fold["mcc"]["mean"] > 0.0
        and fold["macro_f1"]["mean"] > fold["train_majority_macro_f1"]["mean"]
        for fold in fold_aggregates
    ]
    stable = (
        sum(positive) >= 3
        and overall["balanced_accuracy"]["mean"] > 1.0 / 3.0
        and overall["mcc"]["mean"] > 0.0
        and overall["macro_f1"]["mean"]
        > overall["train_majority_macro_f1"]["mean"]
    )
    return positive, stable


def select_classification_label(
    label_results: dict[str, dict[str, object]],
) -> str | None:
    passing = [
        method
        for method in LABEL_METHODS
        if bool(label_results[method]["CLASSIFICATION_TEMPORAL_STABLE"])
    ]
    if not passing:
        return None

    def rank(method: str) -> tuple[float, float, float, float]:
        result = label_results[method]
        aggregate = result["overall_20_evaluation_aggregate"]
        if not isinstance(aggregate, dict):
            raise TypeError("Label aggregate must be a mapping")
        return (
            float(result["classification_positive_fold_count"]),
            float(aggregate["balanced_accuracy"]["mean"]),
            float(aggregate["macro_f1"]["mean"]),
            float(aggregate["mcc"]["mean"]),
        )

    ordered = sorted(passing, key=rank, reverse=True)
    if len(ordered) > 1 and rank(ordered[0]) == rank(ordered[1]):
        raise ValueError("Frozen label selection criteria produce an unresolved tie")
    return ordered[0]


def _scope_summary(data: ClassificationData, positions: np.ndarray) -> dict[str, object]:
    times = data.samples.decision_times[positions]
    return {
        "sample_count": len(times),
        "first_decision_time": times[0].isoformat(),
        "last_decision_time": times[-1].isoformat(),
        "sample_identity_sha256": _sample_identity(times),
    }


def _temporal_design_report(
    data: ClassificationData,
    blocks: list[TemporalBlock],
    folds: list[OuterFold],
) -> dict[str, object]:
    return {
        "common_anchor_count": len(data.samples.decision_times),
        "common_anchor_identity_sha256": _sample_identity(data.samples.decision_times),
        "blocks": [
            {"block": f"B{block.number}", **_scope_summary(data, block.positions)}
            for block in blocks
        ],
        "folds": [
            {
                "fold": fold.number,
                "outer_pool": _scope_summary(data, fold.outer_pool_positions),
                "inner_train": _scope_summary(data, fold.inner_train_positions),
                "inner_validation": _scope_summary(data, fold.inner_validation_positions),
                "outer_evaluation": _scope_summary(data, fold.outer_evaluation_positions),
                "purged_outer_boundary_count": fold.purged_outer_boundary_count,
                "purged_inner_boundary_count": fold.purged_inner_boundary_count,
            }
            for fold in folds
        ],
        "target_boundary_purge_hours": 1,
        "paired_across_label_methods": True,
        "original_validation": "NOT READ OR USED",
        "test": "NOT READ OR USED",
    }


def run_lstm_classification_walkforward(*, project_root: Path) -> LSTMRunResult:
    root = project_root.resolve()
    validate_e05_label_audit(root)
    resolved = resolve_classification_configuration(root)
    device_info = require_official_cuda()
    data = load_classification_data(project_root=root)
    blocks, folds = build_classification_temporal_design(data.samples.decision_times)
    design = _temporal_design_report(data, blocks, folds)
    model_config = resolved["model"]
    training_config = resolved["training"]
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise TypeError("Resolved E06 model/training configuration must be mappings")

    trained: list[LabelFoldSeedTraining] = []
    label_results: dict[str, dict[str, object]] = {}
    for method in LABEL_METHODS:
        fold_results: list[dict[str, object]] = []
        all_metrics: list[dict[str, object]] = []
        for fold in folds:
            members = [
                train_label_fold_seed(
                    method=method,
                    fold=fold,
                    seed=seed,
                    data=data,
                    model_config=model_config,
                    training_config=training_config,
                    device=device_info.device,
                )
                for seed in FROZEN_SEEDS
            ]
            trained.extend(members)
            per_seed = [member.metrics for member in members]
            aggregate = aggregate_classification_metrics(per_seed)
            all_metrics.extend(per_seed)
            fold_results.append(
                {
                    "fold": fold.number,
                    "outer_evaluation": _scope_summary(
                        data,
                        fold.outer_evaluation_positions,
                    ),
                    "per_seed": per_seed,
                    "five_seed_aggregate": aggregate,
                }
            )
        overall = aggregate_classification_metrics(all_metrics)
        fold_aggregates = [result["five_seed_aggregate"] for result in fold_results]
        positive, stable = classification_stability_gate(fold_aggregates, overall)  # type: ignore[arg-type]
        for result, flag in zip(fold_results, positive, strict=True):
            result["CLASSIFICATION_POSITIVE"] = flag
        label_results[method] = {
            "label_method": method,
            "fold_results": fold_results,
            "overall_20_evaluation_aggregate": overall,
            "classification_positive_fold_count": sum(positive),
            "CLASSIFICATION_TEMPORAL_STABLE": stable,
        }
    winner = select_classification_label(label_results)
    paired_deltas: list[dict[str, object]] = []
    delta_metrics = (
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "accuracy",
        "log_loss",
        "multiclass_brier",
    )
    for fold_index in range(FOLD_COUNT):
        l1 = label_results["L1_FIXED"]["fold_results"][fold_index]["five_seed_aggregate"]  # type: ignore[index]
        l2 = label_results["L2_ATR_ADAPTIVE"]["fold_results"][fold_index]["five_seed_aggregate"]  # type: ignore[index]
        paired_deltas.append(
            {
                "fold": fold_index + 1,
                "L2_minus_L1_five_seed_mean": {
                    metric: float(l2[metric]["mean"] - l1[metric]["mean"])
                    for metric in delta_metrics
                },
            }
        )
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does ATR-adaptive labeling improve paired out-of-time three-class prediction?",
        "compared_labels": list(LABEL_METHODS),
        "excluded_label": {
            "method": "L3_ATR_HYSTERESIS_3",
            "reason": "E05-L-A SEMANTIC_DEGRADATION=true",
        },
        "temporal_design": design,
        "frozen_seeds": list(FROZEN_SEEDS),
        "label_results": label_results,
        "paired_L2_minus_L1_diagnostics": paired_deltas,
        "CLASSIFICATION_LABEL_WINNER": "NONE" if winner is None else winner,
        "iid_significance_tests": "NOT COMPUTED",
        "profitability_backtest": "NOT PERFORMED",
        "original_validation": "NOT READ OR USED",
        "test_set": "NOT READ OR USED",
        "training": {
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "model": model_config,
            "training": training_config,
        },
    }

    run_id = f"{EXPERIMENT_ID}_1h_F0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    metrics_path = run_directory / "metrics.json"
    history_path = run_directory / "training_history.json"
    resolved_path = run_directory / "resolved_config.json"
    predictions_path = run_directory / "predictions.parquet"
    _write_json(metrics_path, result)
    _write_json(
        history_path,
        {
            f"{member.label_method}_fold_{member.fold}_seed_{member.seed}": {
                "inner_selection": member.inner_history,
                "outer_refit": member.refit_history,
            }
            for member in trained
        },
    )
    _write_json(
        resolved_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "compared_labels": list(LABEL_METHODS),
            "frozen_seeds": list(FROZEN_SEEDS),
            "temporal_design": design,
            **resolved,
        },
    )
    prediction_frames = []
    for member in trained:
        positions = folds[member.fold - 1].outer_evaluation_positions
        prediction_frames.append(
            pd.DataFrame(
                {
                    "decision_time": data.samples.decision_times[positions],
                    "fold": member.fold,
                    "seed": member.seed,
                    "label_method": member.label_method,
                    "true_class": member.true_classes,
                    "predicted_class": member.predicted_classes,
                    "probability_down": member.probabilities[:, 0],
                    "probability_neutral": member.probabilities[:, 1],
                    "probability_up": member.probabilities[:, 2],
                    "raw_future_return": member.raw_future_returns,
                }
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    predictions.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)
    return LSTMRunResult(metrics_path=metrics_path, run_directory=run_directory, result=result)
