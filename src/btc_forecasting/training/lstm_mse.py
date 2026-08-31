from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm import (
    LOOKBACK_HOURS,
    LSTMRunResult,
    SequenceSamples,
    _load_resolved_configuration,
    _write_json,
    configure_determinism,
    evaluate_validation,
    fit_lstm,
    fit_train_feature_scaler,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)

ABLATION_CONFIG_RELATIVE_PATH = Path("configs/ablations/e03_mse.yaml")
EXPECTED_CHANGED_PATHS = (
    "early_stopping.monitor",
    "loss.delta",
    "loss.type",
)


@dataclass(frozen=True)
class ResolvedMSEConfiguration:
    feature: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    base_experiment: dict[str, Any]
    ablation_experiment: dict[str, Any]


def _changed_leaf_paths(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    changes: list[str] = []
    for key in sorted(set(baseline) | set(candidate)):
        path = f"{prefix}.{key}" if prefix else key
        left = baseline.get(key)
        right = candidate.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.extend(_changed_leaf_paths(left, right, prefix=path))
        elif left != right:
            changes.append(path)
    return changes


def resolve_mse_configuration(*, project_root: Path) -> ResolvedMSEConfiguration:
    root = project_root.resolve()
    feature, model, base_training, base_experiment = _load_resolved_configuration(root)
    document = load_yaml(root / ABLATION_CONFIG_RELATIVE_PATH)
    ablation = document["experiment"]
    overrides = document["training_overrides"]
    required_identity = {
        "id": "E03-MSE",
        "base_experiment": "E03",
        "evaluation_split": "validation",
        "seed": 42,
        "changed_modeling_choice": "loss_only",
    }
    mismatches = {
        key: ablation.get(key)
        for key, expected in required_identity.items()
        if ablation.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid E03-MSE experiment identity: {mismatches}")

    training = copy.deepcopy(base_training)
    training["loss"] = copy.deepcopy(overrides["loss"])
    training["early_stopping"]["monitor"] = overrides["early_stopping_monitor"]
    changed_paths = tuple(_changed_leaf_paths(base_training, training))
    if changed_paths != EXPECTED_CHANGED_PATHS:
        raise ValueError(f"E03-MSE must be a loss-only ablation, got {changed_paths}")
    if training["loss"] != {
        "type": "torch.nn.MSELoss",
        "reduction": "mean",
        "target_scale": "unscaled_one_hour_log_return",
    }:
        raise ValueError("E03-MSE loss must be unscaled mean-reduction MSE")
    if training["early_stopping"]["monitor"] != "validation_mse_loss":
        raise ValueError("E03-MSE early stopping must monitor validation MSE")
    return ResolvedMSEConfiguration(
        feature=feature,
        model=model,
        training=training,
        base_experiment=base_experiment,
        ablation_experiment=ablation,
    )


def validation_magnitude_diagnostics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Magnitude diagnostic inputs must have the same non-empty shape")
    target_std = float(np.std(truth, ddof=0))
    prediction_std = float(np.std(predicted, ddof=0))
    mean_abs_target = float(np.mean(np.abs(truth)))
    return {
        "target_mean": float(np.mean(truth)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": prediction_std,
        "prediction_std_over_target_std": (
            float(prediction_std / target_std) if target_std != 0.0 else float("nan")
        ),
        "mean_abs_prediction_over_mean_abs_target": (
            float(np.mean(np.abs(predicted)) / mean_abs_target)
            if mean_abs_target != 0.0
            else float("nan")
        ),
        "prediction_positive_ratio": float(np.mean(predicted > 0.0)),
        "target_positive_ratio": float(np.mean(truth > 0.0)),
    }


def evaluate_mse_ablation(
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    predictions: np.ndarray,
) -> dict[str, object]:
    result = evaluate_validation(
        validation=validation,
        predictions=predictions,
        train=train,
    )
    result["experiment_id"] = "E03-MSE"
    result["model"] = "lstm_mse_loss_ablation"
    result["controlled_difference"] = {
        "from": "torch.nn.HuberLoss(delta=0.01, reduction=mean)",
        "to": "torch.nn.MSELoss(reduction=mean)",
        "early_stopping_monitor": "validation_mse_loss",
    }
    result["validation_magnitude_diagnostics"] = validation_magnitude_diagnostics(
        validation.targets,
        predictions,
    )
    return result


def run_lstm_mse_ablation(*, project_root: Path) -> LSTMRunResult:
    """Execute the CUDA-only E03-MSE controlled loss ablation."""
    root = project_root.resolve()
    resolved = resolve_mse_configuration(project_root=root)
    device_info = require_official_cuda()
    seed = int(resolved.ablation_experiment["seed"])
    configure_determinism(seed)

    canonical_path = root / CANONICAL_1H_RELATIVE_PATH
    target_path = root / TARGET_RELATIVE_PATH
    split_path = root / FROZEN_SPLIT_RELATIVE_PATH
    prepared = load_sequence_development_samples(
        canonical_path=canonical_path,
        target_path=target_path,
        split_metadata_path=split_path,
    )
    scaled = fit_train_feature_scaler(prepared.train, prepared.validation)
    model = LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(resolved.model["hidden_size"]),
        num_layers=int(resolved.model["num_layers"]),
        configured_dropout=float(resolved.model["dropout"]),
    )
    outcome = fit_lstm(
        model,
        train=scaled.train,
        validation=scaled.validation,
        training_config=resolved.training,
        seed=seed,
        device=device_info.device,
    )
    validation_predictions = predict_lstm(
        outcome.model,
        scaled.validation.features,
        batch_size=int(resolved.training["batch_size"]),
        device=device_info.device,
    )
    result = evaluate_mse_ablation(
        train=scaled.train,
        validation=scaled.validation,
        predictions=validation_predictions,
    )
    result["training"] = {
        "device": str(device_info.device),
        "gpu_name": device_info.gpu_name,
        "seed": seed,
        "epochs_trained": outcome.epochs_trained,
        "best_epoch": outcome.best_epoch,
        "best_validation_mse_loss": outcome.best_validation_loss,
        "duration_seconds": outcome.duration_seconds,
        "configured_dropout": model.configured_dropout,
        "effective_lstm_dropout": model.effective_lstm_dropout,
    }

    run_id = f"E03-MSE_1h_F0_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.parquet"
    history_path = run_directory / "training_history.json"
    resolved_config_path = run_directory / "resolved_config.json"
    checkpoint_path = model_directory / "lstm.pt"
    scaler_path = scaler_directory / "feature_scaler.joblib"

    manifest = create_run_manifest(
        project_root=root,
        experiment_id="E03-MSE",
        run_id=run_id,
    )
    manifest.update(
        {
            "base_experiment_id": "E03",
            "controlled_difference": result["controlled_difference"],
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "seed": seed,
            "configured_dropout": model.configured_dropout,
            "effective_lstm_dropout": model.effective_lstm_dropout,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "feature_set": resolved.feature["id"],
            "feature_names": list(F0_FEATURE_NAMES),
            "input_shape": ["batch", LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
            "evaluated_splits": ["validation"],
            "test_set": "NOT EVALUATED",
            "data": {
                "canonical_1h": {
                    "path": CANONICAL_1H_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(canonical_path),
                },
                "target": {
                    "path": TARGET_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(target_path),
                },
            },
            "split": {
                "path": FROZEN_SPLIT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(split_path),
            },
            "artifacts": {
                "metrics": metrics_path.relative_to(root).as_posix(),
                "predictions": predictions_path.relative_to(root).as_posix(),
                "training_history": history_path.relative_to(root).as_posix(),
                "resolved_config": resolved_config_path.relative_to(root).as_posix(),
                "checkpoint": checkpoint_path.relative_to(root).as_posix(),
                "feature_scaler": scaler_path.relative_to(root).as_posix(),
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    _write_json(metrics_path, result)
    _write_json(history_path, outcome.history)
    _write_json(
        resolved_config_path,
        {
            "experiment": resolved.ablation_experiment,
            "base_experiment": resolved.base_experiment,
            "feature_set": resolved.feature,
            "model": resolved.model,
            "training": resolved.training,
            "controlled_difference": result["controlled_difference"],
        },
    )
    prediction_frame = pd.DataFrame(
        {
            "decision_time": scaled.validation.decision_times,
            TARGET_COLUMN: scaled.validation.targets,
            "lstm_mse_prediction": validation_predictions,
            "zero_return_prediction": zero_return_prediction(len(validation_predictions)),
        }
    )
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    prediction_frame.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)

    model_directory.mkdir(parents=True, exist_ok=False)
    scaler_directory.mkdir(parents=True, exist_ok=False)
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": outcome.model.state_dict(),
            "best_epoch": outcome.best_epoch,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "feature_names": list(F0_FEATURE_NAMES),
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    joblib.dump(scaled.feature_scaler, scaler_path)
    return LSTMRunResult(
        metrics_path=metrics_path,
        run_directory=run_directory,
        result=result,
    )
