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
    _write_json,
    configure_determinism,
    evaluate_validation,
    fit_lstm,
    fit_train_feature_scaler,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_mse import (
    ResolvedMSEConfiguration,
    _changed_leaf_paths,
    resolve_mse_configuration,
    validation_magnitude_diagnostics,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
    VOLATILITY_FEATURE_NAME,
)

EXPERIMENT_ID = "E03-VN-MSE"
NORMALIZED_TARGET_SCALE = "causal_rolling_volatility_24h_normalized_return"


@dataclass(frozen=True)
class ResolvedVNMSEConfiguration:
    feature: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    base_experiment: dict[str, Any]
    experiment: dict[str, Any]


@dataclass(frozen=True)
class VolatilityNormalizedSamples:
    raw: SequenceSamples
    normalized: SequenceSamples
    sigma: np.ndarray
    exclusion_count: int
    original_eligible_count: int


def resolve_vn_mse_configuration(*, project_root: Path) -> ResolvedVNMSEConfiguration:
    """Reuse E03-MSE and change only its target representation metadata."""
    baseline: ResolvedMSEConfiguration = resolve_mse_configuration(
        project_root=project_root
    )
    training = copy.deepcopy(baseline.training)
    training["loss"]["target_scale"] = NORMALIZED_TARGET_SCALE
    changed_paths = _changed_leaf_paths(baseline.training, training)
    if changed_paths != ["loss.target_scale"]:
        raise ValueError(
            f"E03-VN-MSE must change only the target representation, got {changed_paths}"
        )
    if training["loss"]["type"] != "torch.nn.MSELoss":
        raise ValueError("E03-VN-MSE must retain E03-MSE loss")
    if training["loss"]["reduction"] != "mean":
        raise ValueError("E03-VN-MSE must retain mean-reduction MSE")
    if training["early_stopping"]["monitor"] != "validation_mse_loss":
        raise ValueError("E03-VN-MSE must retain validation-MSE early stopping")
    return ResolvedVNMSEConfiguration(
        feature=baseline.feature,
        model=baseline.model,
        training=training,
        base_experiment=baseline.ablation_experiment,
        experiment={
            "id": EXPERIMENT_ID,
            "name": "lstm_1h_f0_volatility_normalized_mse_target",
            "base_experiment": "E03-MSE",
            "evaluation_split": "validation",
            "seed": int(baseline.ablation_experiment["seed"]),
            "changed_modeling_choice": "target_representation_only",
        },
    )


def _replace_samples(
    samples: SequenceSamples,
    *,
    mask: np.ndarray,
    targets: np.ndarray,
) -> SequenceSamples:
    return SequenceSamples(
        features=samples.features[mask],
        targets=np.asarray(targets, dtype=np.float64),
        decision_times=samples.decision_times[mask],
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def prepare_volatility_normalized_samples(
    samples: SequenceSamples,
) -> VolatilityNormalizedSamples:
    """Build z targets from the causal endpoint sigma without adding an epsilon."""
    if samples.features.ndim != 3 or samples.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("E03 samples do not match the frozen F0 sequence shape")
    raw_targets = np.asarray(samples.targets, dtype=np.float64)
    if not np.isfinite(raw_targets).all():
        raise ValueError("Official raw targets must be finite")
    sigma = np.asarray(
        samples.features[:, -1, VOLATILITY_FEATURE_INDEX],
        dtype=np.float64,
    )
    valid = np.isfinite(sigma) & (sigma > 0.0)
    eligible_targets = raw_targets[valid]
    eligible_sigma = sigma[valid]
    normalized_targets = eligible_targets / eligible_sigma
    if len(normalized_targets) == 0:
        raise ValueError("No samples have a valid causal endpoint volatility")
    if not np.isfinite(normalized_targets).all():
        raise ValueError("Volatility-normalized targets must be finite")
    return VolatilityNormalizedSamples(
        raw=_replace_samples(samples, mask=valid, targets=eligible_targets),
        normalized=_replace_samples(samples, mask=valid, targets=normalized_targets),
        sigma=eligible_sigma,
        exclusion_count=int(np.count_nonzero(~valid)),
        original_eligible_count=int(len(samples.targets)),
    )


def reconstruct_raw_predictions(
    normalized_predictions: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(normalized_predictions, dtype=np.float64)
    volatility = np.asarray(sigma, dtype=np.float64)
    if predictions.shape != volatility.shape or predictions.size == 0:
        raise ValueError("Normalized predictions and sigma must have equal non-empty shape")
    if not np.isfinite(predictions).all():
        raise ValueError("Normalized predictions must be finite")
    if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
        raise ValueError("Prediction reconstruction requires finite positive sigma")
    return predictions * volatility


def normalized_space_diagnostics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Normalized diagnostics require equal non-empty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Normalized diagnostics require finite values")
    return {
        "z_target_mean": float(np.mean(truth)),
        "z_target_std": float(np.std(truth, ddof=0)),
        "z_prediction_mean": float(np.mean(predicted)),
        "z_prediction_std": float(np.std(predicted, ddof=0)),
    }


def evaluate_vn_mse_experiment(
    *,
    train: VolatilityNormalizedSamples,
    validation: VolatilityNormalizedSamples,
    normalized_predictions: np.ndarray,
) -> dict[str, object]:
    raw_predictions = reconstruct_raw_predictions(
        normalized_predictions,
        validation.sigma,
    )
    result = evaluate_validation(
        train=train.raw,
        validation=validation.raw,
        predictions=raw_predictions,
    )
    identity = str(result["lstm_validation"]["sample_identity_sha256"])
    same_validation_identity = validation.exclusion_count == 0
    result["experiment_id"] = EXPERIMENT_ID
    result["model"] = "lstm_volatility_normalized_mse_target"
    result["controlled_difference"] = {
        "baseline": "E03-MSE",
        "from": "raw_target_y_t",
        "to": "z_t = y_t / rolling_volatility_24h_t",
        "inference": "y_hat_t = z_hat_t * rolling_volatility_24h_t",
        "epsilon_added": False,
        "loss": "torch.nn.MSELoss(reduction=mean)",
    }
    result["data"].update(
        {
            "train_pre_normalization_eligible_samples": train.original_eligible_count,
            "validation_pre_normalization_eligible_samples": (
                validation.original_eligible_count
            ),
            "train_invalid_sigma_exclusions": train.exclusion_count,
            "validation_invalid_sigma_exclusions": validation.exclusion_count,
        }
    )
    result["raw_return_prediction_diagnostics"] = validation_magnitude_diagnostics(
        validation.raw.targets,
        raw_predictions,
    )
    result["normalized_space_diagnostics"] = normalized_space_diagnostics(
        validation.normalized.targets,
        normalized_predictions,
    )
    result["comparability_to_e03_mse"] = {
        "validation_n": int(len(validation.raw.targets)),
        "sample_identity_sha256": identity,
        "same_validation_sample_identity_by_construction": same_validation_identity,
        "difference_reason": (
            None
            if same_validation_identity
            else (
                f"{validation.exclusion_count} E03-MSE validation samples were "
                "excluded because endpoint sigma was missing, zero, non-finite, or invalid"
            )
        ),
    }
    return result


def run_lstm_vn_mse_experiment(*, project_root: Path) -> LSTMRunResult:
    """Execute the CUDA-only E03-VN-MSE controlled target experiment."""
    root = project_root.resolve()
    resolved = resolve_vn_mse_configuration(project_root=root)
    device_info = require_official_cuda()
    seed = int(resolved.experiment["seed"])
    configure_determinism(seed)

    canonical_path = root / CANONICAL_1H_RELATIVE_PATH
    target_path = root / TARGET_RELATIVE_PATH
    split_path = root / FROZEN_SPLIT_RELATIVE_PATH
    prepared = load_sequence_development_samples(
        canonical_path=canonical_path,
        target_path=target_path,
        split_metadata_path=split_path,
    )
    train = prepare_volatility_normalized_samples(prepared.train)
    validation = prepare_volatility_normalized_samples(prepared.validation)
    scaled = fit_train_feature_scaler(train.normalized, validation.normalized)

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
    normalized_predictions = predict_lstm(
        outcome.model,
        scaled.validation.features,
        batch_size=int(resolved.training["batch_size"]),
        device=device_info.device,
    )
    raw_predictions = reconstruct_raw_predictions(
        normalized_predictions,
        validation.sigma,
    )
    result = evaluate_vn_mse_experiment(
        train=train,
        validation=validation,
        normalized_predictions=normalized_predictions,
    )
    result["training"] = {
        "device": str(device_info.device),
        "gpu_name": device_info.gpu_name,
        "seed": seed,
        "epochs_trained": outcome.epochs_trained,
        "best_epoch": outcome.best_epoch,
        "best_validation_normalized_target_mse_loss": outcome.best_validation_loss,
        "duration_seconds": outcome.duration_seconds,
        "configured_dropout": model.configured_dropout,
        "effective_lstm_dropout": model.effective_lstm_dropout,
    }

    run_id = f"{EXPERIMENT_ID}_1h_F0_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
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
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
    )
    manifest.update(
        {
            "base_experiment_id": "E03-MSE",
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
            "target_definition": result["controlled_difference"],
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
            "experiment": resolved.experiment,
            "base_experiment": resolved.base_experiment,
            "feature_set": resolved.feature,
            "model": resolved.model,
            "training": resolved.training,
            "controlled_difference": result["controlled_difference"],
        },
    )
    prediction_frame = pd.DataFrame(
        {
            "decision_time": validation.raw.decision_times,
            TARGET_COLUMN: validation.raw.targets,
            VOLATILITY_FEATURE_NAME: validation.sigma,
            "normalized_target": validation.normalized.targets,
            "lstm_normalized_prediction": normalized_predictions,
            "lstm_raw_return_prediction": raw_predictions,
            "zero_return_prediction": zero_return_prediction(len(raw_predictions)),
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
            "experiment_id": EXPERIMENT_ID,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "feature_names": list(F0_FEATURE_NAMES),
            "target_definition": result["controlled_difference"],
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
