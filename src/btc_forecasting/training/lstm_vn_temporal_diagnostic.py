from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import (
    SequenceSamples,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_diagnostics import (
    DiagnosticRunResult,
    _resolve_inside,
    _restore_model,
    _write_diagnostic,
    apply_restored_scaler,
)
from btc_forecasting.training.lstm_generalization import (
    _correlation,
    _sign_consistency,
)
from btc_forecasting.training.lstm_vn_learning_diagnostic import (
    load_vn_mse_source_run,
    normalized_target_metrics,
)
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    prepare_volatility_normalized_samples,
)

TRAIN_BLOCK_COUNT = 6


def summarize_training_history(history: object) -> dict[str, object]:
    if not isinstance(history, list) or not history:
        raise ValueError("E03-VN-TD requires a non-empty saved training history")
    epochs: list[dict[str, float | int]] = []
    for expected_epoch, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise ValueError("Each training-history row must be an object")
        epoch = int(row["epoch"])
        if epoch != expected_epoch:
            raise ValueError("Training-history epochs must be consecutive from one")
        train_mse = float(row["train_mse_loss"])
        validation_mse = float(row["validation_mse_loss"])
        learning_rate = float(row["learning_rate"])
        values = np.array([train_mse, validation_mse, learning_rate])
        if not np.isfinite(values).all():
            raise ValueError("Training-history values must be finite")
        epochs.append(
            {
                "epoch": epoch,
                "train_normalized_target_mse": train_mse,
                "validation_normalized_target_mse": validation_mse,
                "learning_rate": learning_rate,
            }
        )
    best_validation = min(
        epochs,
        key=lambda row: float(row["validation_normalized_target_mse"]),
    )
    lowest_train = min(
        epochs,
        key=lambda row: float(row["train_normalized_target_mse"]),
    )
    final = epochs[-1]
    return {
        "epochs": epochs,
        "best_validation_epoch": int(best_validation["epoch"]),
        "lowest_train_mse_epoch": int(lowest_train["epoch"]),
        "final_train_mse": float(final["train_normalized_target_mse"]),
        "final_validation_mse": float(final["validation_normalized_target_mse"]),
    }


def read_source_training_history(
    *,
    project_root: Path,
    manifest: dict[str, Any],
) -> dict[str, object]:
    root = project_root.resolve()
    history_path = _resolve_inside(
        root,
        Path(manifest["artifacts"]["training_history"]),
    )
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Missing E03-VN-MSE training history: {history_path.relative_to(root)}"
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return summarize_training_history(history)


def train_temporal_block_statistics(
    train: SequenceSamples,
    predictions: np.ndarray,
    *,
    block_count: int = TRAIN_BLOCK_COUNT,
) -> list[dict[str, object]]:
    predicted = np.asarray(predictions, dtype=np.float64)
    if len(predicted) != len(train.targets):
        raise ValueError("Train predictions must align one-to-one with normalized targets")
    if block_count != TRAIN_BLOCK_COUNT:
        raise ValueError("E03-VN-TD requires exactly six chronological train blocks")
    blocks: list[dict[str, object]] = []
    for number, positions in enumerate(
        np.array_split(np.arange(len(predicted)), block_count),
        1,
    ):
        if len(positions) == 0:
            raise ValueError("Training samples are insufficient for six non-empty blocks")
        targets = train.targets[positions]
        block_predictions = predicted[positions]
        metrics = normalized_target_metrics(targets, block_predictions)
        zero = metrics["zero_same_rows"]
        skill = metrics["skill"]
        assert isinstance(zero, dict)
        assert isinstance(skill, dict)
        blocks.append(
            {
                "block": number,
                "start_decision_time": train.decision_times[positions[0]].isoformat(),
                "end_decision_time": train.decision_times[positions[-1]].isoformat(),
                "n": int(len(positions)),
                "target_std": metrics["target_std"],
                "prediction_std": metrics["prediction_std"],
                "prediction_std_over_target_std": metrics[
                    "prediction_std_over_target_std"
                ],
                "pearson_ic": metrics["pearson_ic"],
                "spearman_rank_ic": metrics["spearman_rank_ic"],
                "directional_accuracy": metrics["directional_accuracy"],
                "model_mse": metrics["mse"],
                "zero_same_rows_mse": zero["mse"],
                "mse_skill": skill["mse"],
            }
        )
    return blocks


def normalized_feature_signal_stability(
    train: VolatilityNormalizedSamples,
    validation: VolatilityNormalizedSamples,
) -> dict[str, object]:
    if train.normalized.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("Training sequences do not match frozen F0 features")
    if validation.normalized.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("Validation sequences do not match frozen F0 features")
    result: dict[str, object] = {}
    for index, name in enumerate(F0_FEATURE_NAMES):
        train_spearman = _correlation(
            train.normalized.features[:, -1, index],
            train.normalized.targets,
            rank=True,
        )
        validation_spearman = _correlation(
            validation.normalized.features[:, -1, index],
            validation.normalized.targets,
            rank=True,
        )
        result[name] = {
            "train_spearman": train_spearman,
            "validation_spearman": validation_spearman,
            "sign_consistent": _sign_consistency(
                train_spearman,
                validation_spearman,
            ),
        }
    return result


def build_temporal_diagnostic_report(
    *,
    source_run_id: str,
    training_history: dict[str, object],
    train_blocks: list[dict[str, object]],
    feature_stability: dict[str, object],
    train_invalid_sigma_exclusions: int,
    validation_invalid_sigma_exclusions: int,
    device: str,
    gpu_name: str,
) -> dict[str, object]:
    return {
        "diagnostic_id": "E03-VN-TD",
        "diagnostic_type": "training_and_temporal_stability",
        "source_run_id": source_run_id,
        "device": device,
        "gpu_name": gpu_name,
        "training_history": training_history,
        "train_temporal_blocks": train_blocks,
        "normalized_target_endpoint_feature_signal_stability": feature_stability,
        "normalization_exclusions": {
            "train": train_invalid_sigma_exclusions,
            "validation": validation_invalid_sigma_exclusions,
        },
        "evaluated_splits": ["train", "validation"],
        "validation_usage": "endpoint_feature_signal_only",
        "test_set": "NOT EVALUATED",
    }


def run_vn_temporal_diagnostic(
    *,
    project_root: Path,
    source_run: Path,
) -> DiagnosticRunResult:
    root = project_root.resolve()
    source = load_vn_mse_source_run(project_root=root, source_run=source_run)
    device_info = require_official_cuda()
    training_history = read_source_training_history(
        project_root=root,
        manifest=source.manifest,
    )
    prepared = load_sequence_development_samples(
        canonical_path=source.canonical_path,
        target_path=source.target_path,
        split_metadata_path=source.split_path,
    )
    train = prepare_volatility_normalized_samples(prepared.train)
    validation = prepare_volatility_normalized_samples(prepared.validation)
    scaler = joblib.load(source.scaler_path)
    if not isinstance(scaler, RobustScaler):
        raise TypeError("E03-VN-TD source scaler is not a RobustScaler")
    scaled_train = apply_restored_scaler(train.normalized, scaler)
    model = _restore_model(source, device=device_info.device)
    train_predictions = predict_lstm(
        model,
        scaled_train.features,
        batch_size=int(source.training_config["batch_size"]),
        device=device_info.device,
    )
    result = build_temporal_diagnostic_report(
        source_run_id=str(source.manifest["run_id"]),
        training_history=training_history,
        train_blocks=train_temporal_block_statistics(
            train.normalized,
            train_predictions,
        ),
        feature_stability=normalized_feature_signal_stability(
            train,
            validation,
        ),
        train_invalid_sigma_exclusions=train.exclusion_count,
        validation_invalid_sigma_exclusions=validation.exclusion_count,
        device=str(device_info.device),
        gpu_name=device_info.gpu_name,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03VNTD_temporal_{timestamp}",
        result=result,
    )
