from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.baselines.ridge import (
    LOOKBACK_HOURS,
    RidgeRunResult,
    RidgeSamples,
    TrainedRidge,
    _load_resolved_configuration,
    _sample_identity,
    _write_json,
    fit_ridge_model,
    load_development_samples,
)
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm_vn_mse import reconstruct_raw_predictions
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
    VOLATILITY_FEATURE_NAME,
)

EXPERIMENT_ID = "E03-VN-Ridge"
EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY = (
    "3907f80c1b59c98d13d1a733953e7a679b528747b953d3e9db58c63cc10ba13c"
)
ENDPOINT_VOLATILITY_COLUMN = (
    (LOOKBACK_HOURS - 1) * len(F0_FEATURE_NAMES) + VOLATILITY_FEATURE_INDEX
)


@dataclass(frozen=True)
class VolatilityNormalizedRidgeSamples:
    raw: RidgeSamples
    normalized: RidgeSamples
    sigma: np.ndarray
    exclusion_count: int
    original_eligible_count: int


@dataclass(frozen=True)
class VNRidgeEvaluation:
    result: dict[str, object]
    normalized_predictions: np.ndarray
    raw_predictions: np.ndarray


def _replace_samples(
    samples: RidgeSamples,
    *,
    mask: np.ndarray,
    targets: np.ndarray,
) -> RidgeSamples:
    return RidgeSamples(
        features=samples.features[mask],
        targets=np.asarray(targets, dtype=np.float64),
        decision_times=samples.decision_times[mask],
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def prepare_volatility_normalized_ridge_samples(
    samples: RidgeSamples,
) -> VolatilityNormalizedRidgeSamples:
    expected_width = LOOKBACK_HOURS * len(F0_FEATURE_NAMES)
    if samples.features.ndim != 2 or samples.features.shape[1] != expected_width:
        raise ValueError(f"Expected flattened Ridge windows with width {expected_width}")
    raw_targets = np.asarray(samples.targets, dtype=np.float64)
    if not np.isfinite(raw_targets).all():
        raise ValueError("Official raw Ridge targets must be finite")
    sigma = np.asarray(
        samples.features[:, ENDPOINT_VOLATILITY_COLUMN],
        dtype=np.float64,
    )
    valid = np.isfinite(sigma) & (sigma > 0.0)
    eligible_targets = raw_targets[valid]
    eligible_sigma = sigma[valid]
    normalized_targets = eligible_targets / eligible_sigma
    if len(normalized_targets) == 0:
        raise ValueError("No Ridge samples have valid causal endpoint volatility")
    if not np.isfinite(normalized_targets).all():
        raise ValueError("Normalized Ridge targets must be finite")
    return VolatilityNormalizedRidgeSamples(
        raw=_replace_samples(samples, mask=valid, targets=eligible_targets),
        normalized=_replace_samples(samples, mask=valid, targets=normalized_targets),
        sigma=eligible_sigma,
        exclusion_count=int(np.count_nonzero(~valid)),
        original_eligible_count=int(len(samples.targets)),
    )


def _finite_float(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def evaluate_vn_ridge_control(
    trained: TrainedRidge,
    *,
    train: VolatilityNormalizedRidgeSamples,
    validation: VolatilityNormalizedRidgeSamples,
    expected_validation_identity: str = EXPECTED_E03_VN_MSE_VALIDATION_IDENTITY,
) -> VNRidgeEvaluation:
    normalized_predictions = trained.predict(validation.normalized.features)
    raw_predictions = reconstruct_raw_predictions(
        normalized_predictions,
        validation.sigma,
    )
    identity = _sample_identity(validation.raw.decision_times)
    if identity != expected_validation_identity:
        raise ValueError(
            "E03-VN-Ridge validation identity differs from E03-VN-MSE: "
            f"{identity} != {expected_validation_identity}"
        )
    raw_metrics = regression_metrics(validation.raw.targets, raw_predictions)
    zero_metrics = regression_loss_metrics(
        validation.raw.targets,
        zero_return_prediction(len(validation.raw.targets)),
    )
    normalized_metrics = regression_metrics(
        validation.normalized.targets,
        normalized_predictions,
    )
    normalized_mse = float(
        np.mean(np.square(validation.normalized.targets - normalized_predictions))
    )
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "model": "ridge_volatility_normalized_target_control",
        "controlled_difference": {
            "base_implementation": "E02",
            "comparison_model": "E03-VN-MSE",
            "from": "raw_target_y_t",
            "to": "z_t = y_t / rolling_volatility_24h_t",
            "inference": "y_hat_t = z_hat_t * rolling_volatility_24h_t",
            "epsilon_added": False,
        },
        "evaluated_splits": ["validation"],
        "data": {
            "lookback_hours": LOOKBACK_HOURS,
            "window_shape": [LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
            "flattened_feature_count": LOOKBACK_HOURS * len(F0_FEATURE_NAMES),
            "window_flattening_order": "oldest_to_newest_then_f0_config_order",
            "train_pre_normalization_eligible_samples": train.original_eligible_count,
            "train_eligible_samples": len(train.normalized.targets),
            "train_invalid_sigma_exclusions": train.exclusion_count,
            "train_excluded_insufficient_or_nonconsecutive_lookback": (
                train.raw.excluded_lookback_count
            ),
            "validation_pre_normalization_eligible_samples": (
                validation.original_eligible_count
            ),
            "validation_eligible_samples": len(validation.normalized.targets),
            "validation_invalid_sigma_exclusions": validation.exclusion_count,
            "validation_excluded_insufficient_or_nonconsecutive_lookback": (
                validation.raw.excluded_lookback_count
            ),
        },
        "raw_return_validation": {
            "n": int(raw_metrics["n"]),
            "mae": _finite_float(raw_metrics["mae"]),
            "rmse": _finite_float(raw_metrics["rmse"]),
            "r2": _finite_float(raw_metrics["r2"]),
            "pearson_ic": _finite_float(raw_metrics["pearson_ic"]),
            "spearman_rank_ic": _finite_float(raw_metrics["spearman_rank_ic"]),
            "directional_accuracy": _finite_float(
                raw_metrics["directional_accuracy"]
            ),
            "sample_identity_sha256": identity,
        },
        "zero_return_same_validation_rows": {
            "n": int(zero_metrics["n"]),
            "mae": zero_mae,
            "rmse": zero_rmse,
            "sample_identity_sha256": identity,
        },
        "skill": {
            "mae": 1.0 - float(raw_metrics["mae"]) / zero_mae,
            "rmse": 1.0 - float(raw_metrics["rmse"]) / zero_rmse,
        },
        "normalized_space_validation": {
            "n": int(normalized_metrics["n"]),
            "mse": normalized_mse,
            "pearson_ic": _finite_float(normalized_metrics["pearson_ic"]),
            "spearman_rank_ic": _finite_float(
                normalized_metrics["spearman_rank_ic"]
            ),
            "prediction_std": float(np.std(normalized_predictions, ddof=0)),
            "target_std": float(np.std(validation.normalized.targets, ddof=0)),
        },
        "e03_vn_mse_sample_identity": {
            "expected": expected_validation_identity,
            "actual": identity,
            "matches": True,
        },
        "test_set": "NOT EVALUATED",
    }
    return VNRidgeEvaluation(
        result=result,
        normalized_predictions=normalized_predictions,
        raw_predictions=raw_predictions,
    )


def run_vn_ridge_control(*, project_root: Path) -> RidgeRunResult:
    root = project_root.resolve()
    canonical_path = root / CANONICAL_1H_RELATIVE_PATH
    target_path = root / TARGET_RELATIVE_PATH
    split_path = root / FROZEN_SPLIT_RELATIVE_PATH
    feature_config, model_config = _load_resolved_configuration(root)
    samples = load_development_samples(
        canonical_path=canonical_path,
        target_path=target_path,
        split_metadata_path=split_path,
    )
    train = prepare_volatility_normalized_ridge_samples(samples.train)
    validation = prepare_volatility_normalized_ridge_samples(samples.validation)
    trained = fit_ridge_model(
        train.normalized,
        alpha=float(model_config["alpha"]),
        fit_intercept=bool(model_config["fit_intercept"]),
    )
    evaluation = evaluate_vn_ridge_control(
        trained,
        train=train,
        validation=validation,
    )

    run_id = f"{EXPERIMENT_ID}_1h_F0_B1_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    predictions_path = run_directory / "predictions.parquet"
    metrics_path = run_directory / "metrics.json"
    model_path = model_directory / "ridge.joblib"
    feature_scaler_path = scaler_directory / "feature_scaler.joblib"
    target_scaler_path = scaler_directory / "target_scaler.joblib"

    manifest = create_run_manifest(
        project_root=root,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
    )
    manifest.update(
        {
            "base_implementation": "E02",
            "comparison_model": "E03-VN-MSE",
            "controlled_difference": evaluation.result["controlled_difference"],
            "model": model_config,
            "timeframe": "1h",
            "lookback": "24h",
            "horizon": "1h",
            "feature_set": feature_config["id"],
            "feature_names": list(F0_FEATURE_NAMES),
            "window_flattening_order": "oldest_to_newest_then_f0_config_order",
            "target_definition": evaluation.result["controlled_difference"],
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
            "row_counts_and_exclusions": evaluation.result["data"],
            "sample_identity": evaluation.result["e03_vn_mse_sample_identity"],
            "artifacts": {
                "metrics": metrics_path.relative_to(root).as_posix(),
                "predictions": predictions_path.relative_to(root).as_posix(),
                "model": model_path.relative_to(root).as_posix(),
                "feature_scaler": feature_scaler_path.relative_to(root).as_posix(),
                "target_scaler": target_scaler_path.relative_to(root).as_posix(),
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    _write_json(metrics_path, evaluation.result)
    prediction_frame = pd.DataFrame(
        {
            "decision_time": validation.raw.decision_times,
            TARGET_COLUMN: validation.raw.targets,
            VOLATILITY_FEATURE_NAME: validation.sigma,
            "normalized_target": validation.normalized.targets,
            "ridge_normalized_prediction": evaluation.normalized_predictions,
            "ridge_raw_return_prediction": evaluation.raw_predictions,
            "zero_return_prediction": zero_return_prediction(
                len(validation.raw.targets)
            ),
        }
    )
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    prediction_frame.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)
    model_directory.mkdir(parents=True, exist_ok=False)
    scaler_directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(trained.model, model_path)
    joblib.dump(trained.feature_scaler, feature_scaler_path)
    joblib.dump(trained.target_scaler, target_scaler_path)
    return RidgeRunResult(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        result=evaluation.result,
    )
