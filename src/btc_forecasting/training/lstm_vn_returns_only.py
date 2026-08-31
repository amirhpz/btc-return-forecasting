from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm import (
    LOOKBACK_HOURS,
    LSTMRunResult,
    ScaledSequenceSamples,
    SequenceSamples,
    _sample_identity,
    _write_json,
    configure_determinism,
    fit_lstm,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    evaluate_vn_mse_experiment,
    prepare_volatility_normalized_samples,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_NAME,
)

EXPERIMENT_ID = "E03-VN-R1"
RETURNS_FEATURE_NAME = "log_return_1h"
RETURNS_FEATURE_INDEX = F0_FEATURE_NAMES.index(RETURNS_FEATURE_NAME)
EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY = (
    "3907f80c1b59c98d13d1a733953e7a679b528747b953d3e9db58c63cc10ba13c"
)


def project_returns_only(samples: SequenceSamples) -> SequenceSamples:
    if samples.features.ndim != 3 or samples.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("Returns-only projection requires frozen 24 x 10 F0 sequences")
    features = samples.features[
        :,
        :,
        RETURNS_FEATURE_INDEX : RETURNS_FEATURE_INDEX + 1,
    ]
    return SequenceSamples(
        features=features.astype(np.float32, copy=True),
        targets=samples.targets,
        decision_times=samples.decision_times,
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def fit_returns_only_feature_scaler(
    train: SequenceSamples,
    validation: SequenceSamples,
) -> ScaledSequenceSamples:
    if train.features.shape[1:] != (LOOKBACK_HOURS, 1):
        raise ValueError("Returns-only train input must have shape [n, 24, 1]")
    if validation.features.shape[1:] != (LOOKBACK_HOURS, 1):
        raise ValueError("Returns-only validation input must have shape [n, 24, 1]")
    if len(train.targets) == 0:
        raise ValueError("Returns-only training samples must not be empty")
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
    )
    train_shape = train.features.shape
    validation_shape = validation.features.shape
    scaled_train = scaler.fit_transform(train.features.reshape(-1, 1)).reshape(
        train_shape
    )
    scaled_validation = scaler.transform(
        validation.features.reshape(-1, 1)
    ).reshape(validation_shape)

    def replace(samples: SequenceSamples, features: np.ndarray) -> SequenceSamples:
        return SequenceSamples(
            features=features.astype(np.float32, copy=False),
            targets=samples.targets,
            decision_times=samples.decision_times,
            candidate_count=samples.candidate_count,
            excluded_lookback_count=samples.excluded_lookback_count,
        )

    return ScaledSequenceSamples(
        train=replace(train, scaled_train),
        validation=replace(validation, scaled_validation),
        feature_scaler=scaler,
    )


def evaluate_returns_only_experiment(
    *,
    train: VolatilityNormalizedSamples,
    validation: VolatilityNormalizedSamples,
    normalized_predictions: np.ndarray,
    expected_validation_identity: str = EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY,
) -> dict[str, object]:
    result = evaluate_vn_mse_experiment(
        train=train,
        validation=validation,
        normalized_predictions=normalized_predictions,
    )
    actual_identity = _sample_identity(validation.raw.decision_times)
    matches = actual_identity == expected_validation_identity
    if matches:
        difference_reason = None
    elif validation.exclusion_count:
        difference_reason = (
            f"{validation.exclusion_count} validation samples were excluded because "
            "endpoint sigma was missing, zero, non-finite, or invalid"
        )
    else:
        difference_reason = (
            "the current source data or frozen split does not yield the frozen "
            "E03-VN-MSE validation decision-time sequence; returns-only projection "
            "itself excludes zero samples"
        )
    diagnostics = result["normalized_space_diagnostics"]
    assert isinstance(diagnostics, dict)
    target_std = float(diagnostics["z_target_std"])
    prediction_std = float(diagnostics["z_prediction_std"])
    normalized_targets = validation.normalized.targets
    normalized_prediction_values = np.asarray(
        normalized_predictions,
        dtype=np.float64,
    )
    normalized_metrics = regression_metrics(
        normalized_targets,
        normalized_prediction_values,
    )
    result["experiment_id"] = EXPERIMENT_ID
    result["model"] = "lstm_volatility_normalized_returns_only"
    result["controlled_difference"] = {
        "baseline": "E03-VN-MSE",
        "from_input": "24 x 10 frozen F0",
        "to_input": "24 x 1 log_return_1h",
        "target_normalization": "z_t = y_t / rolling_volatility_24h_t",
        "inference": "y_hat_t = z_hat_t * rolling_volatility_24h_t",
        "sigma_is_model_input": False,
        "epsilon_added": False,
    }
    data = result["data"]
    assert isinstance(data, dict)
    data["window_shape"] = [LOOKBACK_HOURS, 1]
    data["input_feature_count"] = 1
    data["input_feature_names"] = [RETURNS_FEATURE_NAME]
    data["returns_only_projection_exclusions"] = {
        "train": 0,
        "validation": 0,
    }
    result["normalized_space_validation"] = {
        "n": int(len(normalized_targets)),
        "mse": float(
            np.mean(
                np.square(normalized_targets - normalized_prediction_values)
            )
        ),
        "pearson_ic": float(normalized_metrics["pearson_ic"]),
        "spearman_rank_ic": float(normalized_metrics["spearman_rank_ic"]),
        "prediction_std": prediction_std,
        "target_std": target_std,
        "prediction_std_over_target_std": (
            None if target_std == 0.0 else prediction_std / target_std
        ),
    }
    result["comparability_to_e03_vn_mse"] = {
        "expected_sample_identity_sha256": expected_validation_identity,
        "actual_sample_identity_sha256": actual_identity,
        "matches": matches,
        "difference_reason": difference_reason,
    }
    result.pop("comparability_to_e03_mse", None)
    return result


def run_lstm_vn_returns_only(*, project_root: Path) -> LSTMRunResult:
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
    returns_train = project_returns_only(train.normalized)
    returns_validation = project_returns_only(validation.normalized)
    scaled = fit_returns_only_feature_scaler(returns_train, returns_validation)

    model = LSTMRegressor(
        input_size=1,
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
    result = evaluate_returns_only_experiment(
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

    run_id = f"{EXPERIMENT_ID}_1h_R1_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
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
            "base_experiment_id": "E03-VN-MSE",
            "controlled_difference": result["controlled_difference"],
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "seed": seed,
            "configured_dropout": model.configured_dropout,
            "effective_lstm_dropout": model.effective_lstm_dropout,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "base_feature_set": resolved.feature["id"],
            "input_feature_names": [RETURNS_FEATURE_NAME],
            "input_shape": ["batch", LOOKBACK_HOURS, 1],
            "target_definition": {
                "z_t": "y_t / rolling_volatility_24h_t",
                "raw_inference": "z_hat_t * rolling_volatility_24h_t",
                "epsilon_added": False,
            },
            "sample_identity": result["comparability_to_e03_vn_mse"],
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
            "experiment": {
                "id": EXPERIMENT_ID,
                "base_experiment": "E03-VN-MSE",
                "changed_modeling_choice": "input_features_only",
                "seed": seed,
            },
            "base_experiment": resolved.experiment,
            "base_feature_set": resolved.feature,
            "input_feature_names": [RETURNS_FEATURE_NAME],
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
            "feature_names": [RETURNS_FEATURE_NAME],
            "input_shape": [LOOKBACK_HOURS, 1],
            "target_definition": manifest["target_definition"],
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
