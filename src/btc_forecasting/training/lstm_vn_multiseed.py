from __future__ import annotations

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
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm import (
    LOOKBACK_HOURS,
    LSTMRunResult,
    ScaledSequenceSamples,
    _sample_identity,
    _write_json,
    configure_determinism,
    fit_lstm,
    fit_train_feature_scaler,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_generalization import _correlation
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    prepare_volatility_normalized_samples,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)
from btc_forecasting.training.volatility_normalization import VOLATILITY_FEATURE_NAME

EXPERIMENT_ID = "E03-VN-MS5"
FROZEN_SEEDS = (42, 137, 271, 811, 2027)
PRIMARY_METRIC_PATHS = {
    "mae": ("raw_return_validation", "mae"),
    "rmse": ("raw_return_validation", "rmse"),
    "r2": ("raw_return_validation", "r2"),
    "pearson_ic": ("raw_return_validation", "pearson_ic"),
    "spearman_rank_ic": ("raw_return_validation", "spearman_rank_ic"),
    "directional_accuracy": ("raw_return_validation", "directional_accuracy"),
    "mae_skill": ("skill", "mae"),
    "rmse_skill": ("skill", "rmse"),
    "normalized_mse": ("normalized_space", "mse"),
    "normalized_pearson_ic": ("normalized_space", "pearson_ic"),
    "normalized_spearman_rank_ic": ("normalized_space", "spearman_rank_ic"),
    "normalized_prediction_std": ("normalized_space", "prediction_std"),
    "normalized_target_std": ("normalized_space", "target_std"),
}


@dataclass(frozen=True)
class SeedTraining:
    seed: int
    model: LSTMRegressor
    history: list[dict[str, float | int]]
    normalized_predictions: np.ndarray
    raw_predictions: np.ndarray
    metrics: dict[str, object]


def evaluate_seed(
    *,
    seed: int,
    train: VolatilityNormalizedSamples,
    validation: VolatilityNormalizedSamples,
    normalized_predictions: np.ndarray,
    best_epoch: int,
    epochs_trained: int,
    duration_seconds: float,
) -> dict[str, object]:
    predictions = np.asarray(normalized_predictions, dtype=np.float64)
    raw_predictions = reconstruct_raw_predictions(predictions, validation.sigma)
    raw_metrics = regression_metrics(validation.raw.targets, raw_predictions)
    zero_metrics = regression_loss_metrics(
        validation.raw.targets,
        zero_return_prediction(len(validation.raw.targets)),
    )
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    normalized_targets = validation.normalized.targets
    return {
        "seed": seed,
        "train_sample_count": int(len(train.normalized.targets)),
        "validation_sample_count": int(len(validation.normalized.targets)),
        "train_sample_identity_sha256": _sample_identity(
            train.normalized.decision_times
        ),
        "validation_sample_identity_sha256": _sample_identity(
            validation.normalized.decision_times
        ),
        "raw_return_validation": {
            key: raw_metrics[key]
            for key in (
                "n",
                "mae",
                "rmse",
                "r2",
                "pearson_ic",
                "spearman_rank_ic",
                "directional_accuracy",
            )
        },
        "same_row_zero_return": {
            "n": int(zero_metrics["n"]),
            "mae": zero_mae,
            "rmse": zero_rmse,
        },
        "skill": {
            "mae": 1.0 - float(raw_metrics["mae"]) / zero_mae,
            "rmse": 1.0 - float(raw_metrics["rmse"]) / zero_rmse,
        },
        "normalized_space": {
            "mse": float(np.mean(np.square(predictions - normalized_targets))),
            "pearson_ic": _correlation(normalized_targets, predictions, rank=False),
            "spearman_rank_ic": _correlation(
                normalized_targets,
                predictions,
                rank=True,
            ),
            "prediction_std": float(np.std(predictions, ddof=0)),
            "target_std": float(np.std(normalized_targets, ddof=0)),
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "duration_seconds": duration_seconds,
        },
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
    }


def verify_seed_sample_identities(
    seed_results: list[dict[str, object]],
) -> dict[str, object]:
    seeds = tuple(int(result["seed"]) for result in seed_results)
    if seeds != FROZEN_SEEDS:
        raise ValueError(f"E03-VN-MS5 requires exactly the frozen seeds {FROZEN_SEEDS}")
    train_counts = {int(result["train_sample_count"]) for result in seed_results}
    validation_counts = {
        int(result["validation_sample_count"]) for result in seed_results
    }
    train_identities = {
        str(result["train_sample_identity_sha256"]) for result in seed_results
    }
    validation_identities = {
        str(result["validation_sample_identity_sha256"]) for result in seed_results
    }
    if any(
        len(values) != 1
        for values in (
            train_counts,
            validation_counts,
            train_identities,
            validation_identities,
        )
    ):
        raise ValueError("All five seeds must use identical TRAIN and validation samples")
    return {
        "train_sample_count": train_counts.pop(),
        "validation_sample_count": validation_counts.pop(),
        "train_sample_identity_sha256": train_identities.pop(),
        "validation_sample_identity_sha256": validation_identities.pop(),
        "identical_across_all_five_seeds": True,
    }


def _nested_metric(result: dict[str, object], path: tuple[str, str]) -> float:
    group = result[path[0]]
    if not isinstance(group, dict):
        raise TypeError(f"Metric group is not a mapping: {path[0]}")
    value = float(group[path[1]])
    if not np.isfinite(value):
        raise ValueError(f"Aggregate metric must be finite: {path}")
    return value


def aggregate_seed_metrics(
    seed_results: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    verify_seed_sample_identities(seed_results)
    aggregates: dict[str, dict[str, float]] = {}
    for name, path in PRIMARY_METRIC_PATHS.items():
        values = np.asarray(
            [_nested_metric(result, path) for result in seed_results],
            dtype=np.float64,
        )
        aggregates[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return aggregates


def stability_gate(
    seed_results: list[dict[str, object]],
    aggregates: dict[str, dict[str, float]],
) -> tuple[dict[str, int], bool]:
    verify_seed_sample_identities(seed_results)
    counts = {
        "mae_skill_gt_0": sum(
            _nested_metric(result, ("skill", "mae")) > 0.0
            for result in seed_results
        ),
        "rmse_skill_gt_0": sum(
            _nested_metric(result, ("skill", "rmse")) > 0.0
            for result in seed_results
        ),
        "r2_gt_0": sum(
            _nested_metric(result, ("raw_return_validation", "r2")) > 0.0
            for result in seed_results
        ),
        "pearson_ic_gt_0": sum(
            _nested_metric(result, ("raw_return_validation", "pearson_ic")) > 0.0
            for result in seed_results
        ),
        "spearman_rank_ic_gt_0": sum(
            _nested_metric(
                result,
                ("raw_return_validation", "spearman_rank_ic"),
            )
            > 0.0
            for result in seed_results
        ),
        "directional_accuracy_gt_0_50": sum(
            _nested_metric(
                result,
                ("raw_return_validation", "directional_accuracy"),
            )
            > 0.50
            for result in seed_results
        ),
    }
    stable = (
        all(count >= 4 for count in counts.values())
        and aggregates["mae_skill"]["mean"] > 0.0
        and aggregates["rmse_skill"]["mean"] > 0.0
        and aggregates["r2"]["mean"] > 0.0
    )
    return counts, stable


def build_multiseed_report(
    seed_results: list[dict[str, object]],
) -> dict[str, object]:
    samples = verify_seed_sample_identities(seed_results)
    aggregates = aggregate_seed_metrics(seed_results)
    counts, stable = stability_gate(seed_results, aggregates)
    return {
        "experiment_id": EXPERIMENT_ID,
        "question": "Is the F0 VN-LSTM predictive edge stable across initialization seeds?",
        "experiment_type": "robustness_not_hyperparameter_tuning",
        "frozen_seeds": list(FROZEN_SEEDS),
        "feature_set": "F0",
        "target_definition": {
            "z_t": "future_log_return_1h / rolling_volatility_24h_t",
            "epsilon_added": False,
            "inference": "raw_prediction = normalized_prediction * rolling_volatility_24h_t",
        },
        "samples": samples,
        "per_seed": seed_results,
        "aggregate_primary_metrics": aggregates,
        "positive_condition_seed_counts": counts,
        "stability_gate": {
            "minimum_positive_seeds_per_condition": 4,
            "requires_positive_mean": ["mae_skill", "rmse_skill", "r2"],
            "PRELIMINARY_SEED_STABLE": stable,
            "interpretation": (
                "Seed robustness only; not final model validity or professional forecasting performance."
            ),
        },
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
    }


def run_lstm_vn_multiseed_experiment(*, project_root: Path) -> LSTMRunResult:
    root = project_root.resolve()
    resolved = resolve_vn_mse_configuration(project_root=root)
    device_info = require_official_cuda()
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

    trained: list[SeedTraining] = []
    seed_results: list[dict[str, object]] = []
    for seed in FROZEN_SEEDS:
        configure_determinism(seed)
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
        metrics = evaluate_seed(
            seed=seed,
            train=train,
            validation=validation,
            normalized_predictions=normalized_predictions,
            best_epoch=outcome.best_epoch,
            epochs_trained=outcome.epochs_trained,
            duration_seconds=outcome.duration_seconds,
        )
        seed_results.append(metrics)
        trained.append(
            SeedTraining(
                seed=seed,
                model=outcome.model,
                history=outcome.history,
                normalized_predictions=normalized_predictions,
                raw_predictions=reconstruct_raw_predictions(
                    normalized_predictions,
                    validation.sigma,
                ),
                metrics=metrics,
            )
        )
    result = build_multiseed_report(seed_results)
    result["training"] = {
        "device": str(device_info.device),
        "gpu_name": device_info.gpu_name,
        "configured_dropout": float(resolved.model["dropout"]),
        "effective_lstm_dropout": 0.0,
        "model_config": resolved.model,
        "training_config": resolved.training,
    }

    run_id = f"{EXPERIMENT_ID}_1h_F0_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.parquet"
    history_path = run_directory / "training_history.json"
    resolved_config_path = run_directory / "resolved_config.json"
    scaler_path = scaler_directory / "feature_scaler.joblib"
    checkpoint_paths = {
        seed: model_directory / f"seed_{seed}_lstm.pt" for seed in FROZEN_SEEDS
    }

    manifest = create_run_manifest(
        project_root=root,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
    )
    manifest.update(
        {
            "base_experiment_id": "E03-VN-MSE",
            "experiment_type": "multi_seed_robustness",
            "frozen_seeds": list(FROZEN_SEEDS),
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "feature_set": resolved.feature["id"],
            "feature_names": list(F0_FEATURE_NAMES),
            "input_shape": ["batch", LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
            "samples": result["samples"],
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
                "feature_scaler": scaler_path.relative_to(root).as_posix(),
                "checkpoints": {
                    str(seed): path.relative_to(root).as_posix()
                    for seed, path in checkpoint_paths.items()
                },
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    _write_json(metrics_path, result)
    _write_json(
        history_path,
        {str(member.seed): member.history for member in trained},
    )
    _write_json(
        resolved_config_path,
        {
            "experiment": {
                "id": EXPERIMENT_ID,
                "base_experiment": "E03-VN-MSE",
                "type": "multi_seed_robustness_not_tuning",
                "frozen_seeds": list(FROZEN_SEEDS),
            },
            "feature_set": resolved.feature,
            "model": resolved.model,
            "training": resolved.training,
        },
    )
    prediction_frame = pd.DataFrame(
        {
            "decision_time": validation.raw.decision_times,
            TARGET_COLUMN: validation.raw.targets,
            VOLATILITY_FEATURE_NAME: validation.sigma,
            "normalized_target": validation.normalized.targets,
            "zero_return_prediction": zero_return_prediction(
                len(validation.raw.targets)
            ),
        }
    )
    for member in trained:
        prediction_frame[f"seed_{member.seed}_normalized_prediction"] = (
            member.normalized_predictions
        )
        prediction_frame[f"seed_{member.seed}_raw_return_prediction"] = (
            member.raw_predictions
        )
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    prediction_frame.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)

    model_directory.mkdir(parents=True, exist_ok=False)
    scaler_directory.mkdir(parents=True, exist_ok=False)
    for member in trained:
        checkpoint = checkpoint_paths[member.seed]
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": member.model.state_dict(),
                "experiment_id": EXPERIMENT_ID,
                "seed": member.seed,
                "best_epoch": member.metrics["training"]["best_epoch"],  # type: ignore[index]
                "model_config": resolved.model,
                "training_config": resolved.training,
                "feature_names": list(F0_FEATURE_NAMES),
                "target_definition": "z_t = raw_return / rolling_volatility_24h_t",
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    joblib.dump(scaled.feature_scaler, scaler_path)
    return LSTMRunResult(
        metrics_path=metrics_path,
        run_directory=run_directory,
        result=result,
    )
