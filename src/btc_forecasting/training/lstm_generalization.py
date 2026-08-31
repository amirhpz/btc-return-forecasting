from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import RobustScaler

from btc_forecasting.evaluation.metrics import regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import (
    SequenceSamples,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_diagnostics import (
    DiagnosticRunResult,
    _restore_model,
    _write_diagnostic,
    apply_restored_scaler,
    load_source_run,
)

AUTOCORRELATION_LAGS_HOURS = (1, 3, 6, 12, 24)
VALIDATION_BLOCK_COUNT = 4


def _finite_float(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    result = spearmanr(x, y) if rank else pearsonr(x, y)
    return _finite_float(result.statistic)


def target_regime_statistics(targets: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(targets, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Target regime requires non-empty finite targets")
    quantiles = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "mean_absolute_target": float(np.mean(np.abs(values))),
        "positive_ratio": float(np.mean(values > 0.0)),
        "quantile_05": float(quantiles[0]),
        "quantile_25": float(quantiles[1]),
        "quantile_75": float(quantiles[2]),
        "quantile_95": float(quantiles[3]),
    }


def build_target_regime_report(
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
) -> dict[str, object]:
    train = target_regime_statistics(train_targets)
    validation = target_regime_statistics(validation_targets)
    train_std = float(train["std"])
    train_mean_abs = float(train["mean_absolute_target"])
    return {
        "train": train,
        "validation": validation,
        "validation_over_train": {
            "target_std": (
                float(validation["std"]) / train_std if train_std != 0.0 else None
            ),
            "mean_absolute_target": (
                float(validation["mean_absolute_target"]) / train_mean_abs
                if train_mean_abs != 0.0
                else None
            ),
        },
    }


def validation_block_statistics(
    validation: SequenceSamples,
    predictions: np.ndarray,
    *,
    block_count: int = VALIDATION_BLOCK_COUNT,
) -> list[dict[str, object]]:
    predicted = np.asarray(predictions, dtype=np.float64)
    if len(predicted) != len(validation.targets):
        raise ValueError("Validation predictions must align one-to-one with targets")
    if block_count != VALIDATION_BLOCK_COUNT:
        raise ValueError("E03-G requires exactly four validation blocks")
    blocks: list[dict[str, object]] = []
    for number, positions in enumerate(np.array_split(np.arange(len(predicted)), block_count), 1):
        if len(positions) == 0:
            raise ValueError("Validation is too small for four non-empty blocks")
        targets = validation.targets[positions]
        block_predictions = predicted[positions]
        metrics = regression_metrics(targets, block_predictions)
        blocks.append(
            {
                "block": number,
                "start_decision_time": validation.decision_times[positions[0]].isoformat(),
                "end_decision_time": validation.decision_times[positions[-1]].isoformat(),
                "n": int(len(positions)),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": _finite_float(metrics["r2"]),
                "pearson_ic": _finite_float(metrics["pearson_ic"]),
                "spearman_rank_ic": _finite_float(metrics["spearman_rank_ic"]),
                "directional_accuracy": _finite_float(metrics["directional_accuracy"]),
                "target_std": float(np.std(targets, ddof=0)),
                "prediction_std": float(np.std(block_predictions, ddof=0)),
                "prediction_positive_ratio": float(np.mean(block_predictions > 0.0)),
                "target_positive_ratio": float(np.mean(targets > 0.0)),
            }
        )
    return blocks


def _sign_consistency(train: float | None, validation: float | None) -> bool | None:
    if train is None or validation is None or train == 0.0 or validation == 0.0:
        return None
    return bool(np.sign(train) == np.sign(validation))


def feature_target_stability(
    train: SequenceSamples,
    validation: SequenceSamples,
) -> dict[str, object]:
    if train.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("Training sequences do not match frozen F0 feature count")
    if validation.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("Validation sequences do not match frozen F0 feature count")
    result: dict[str, object] = {}
    for index, name in enumerate(F0_FEATURE_NAMES):
        train_values = train.features[:, -1, index]
        validation_values = validation.features[:, -1, index]
        train_pearson = _correlation(train_values, train.targets, rank=False)
        validation_pearson = _correlation(
            validation_values, validation.targets, rank=False
        )
        train_spearman = _correlation(train_values, train.targets, rank=True)
        validation_spearman = _correlation(
            validation_values, validation.targets, rank=True
        )
        result[name] = {
            "pearson": {
                "train": train_pearson,
                "validation": validation_pearson,
                "sign_consistent": _sign_consistency(
                    train_pearson, validation_pearson
                ),
            },
            "spearman": {
                "train": train_spearman,
                "validation": validation_spearman,
                "sign_consistent": _sign_consistency(
                    train_spearman, validation_spearman
                ),
            },
        }
    return result


def gap_safe_target_autocorrelation(
    samples: SequenceSamples,
    *,
    lags_hours: tuple[int, ...] = AUTOCORRELATION_LAGS_HOURS,
) -> dict[str, object]:
    timestamps = samples.decision_times
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Decision timestamps must be strictly ordered and unique")
    result: dict[str, object] = {}
    for lag in lags_hours:
        if lag <= 0:
            raise ValueError("Autocorrelation lags must be positive")
        earlier_positions = timestamps.get_indexer(timestamps - timedelta(hours=lag))
        current_positions = np.flatnonzero(earlier_positions >= 0)
        matched_earlier = earlier_positions[current_positions]
        result[str(lag)] = {
            "lag_real_hours": lag,
            "n_pairs": int(len(current_positions)),
            "autocorrelation": _correlation(
                samples.targets[current_positions],
                samples.targets[matched_earlier],
                rank=False,
            ),
        }
    return result


def build_generalization_report(
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    validation_predictions: np.ndarray,
    source_run_id: str,
    device: str,
    gpu_name: str,
) -> dict[str, object]:
    return {
        "diagnostic_id": "E03-G",
        "diagnostic_type": "generalization_and_signal",
        "source_run_id": source_run_id,
        "device": device,
        "gpu_name": gpu_name,
        "evaluated_splits": ["train", "validation"],
        "target_regime": build_target_regime_report(
            train.targets, validation.targets
        ),
        "validation_temporal_blocks": validation_block_statistics(
            validation, validation_predictions
        ),
        "f0_endpoint_feature_target_stability": feature_target_stability(
            train, validation
        ),
        "target_return_autocorrelation": {
            "gap_rule": "pairs_require_exact_real_hour_timestamp_difference",
            "train": gap_safe_target_autocorrelation(train),
            "validation": gap_safe_target_autocorrelation(validation),
        },
        "test_set": "NOT EVALUATED",
    }


def run_lstm_generalization_diagnostic(
    *,
    project_root: Path,
    source_run: Path,
) -> DiagnosticRunResult:
    root = project_root.resolve()
    source = load_source_run(project_root=root, source_run=source_run)
    device_info = require_official_cuda()
    prepared = load_sequence_development_samples(
        canonical_path=source.canonical_path,
        target_path=source.target_path,
        split_metadata_path=source.split_path,
    )
    scaler = joblib.load(source.scaler_path)
    if not isinstance(scaler, RobustScaler):
        raise TypeError("E03-G source scaler is not a RobustScaler")
    scaled_validation = apply_restored_scaler(prepared.validation, scaler)
    model = _restore_model(source, device=device_info.device)
    validation_predictions = predict_lstm(
        model,
        scaled_validation.features,
        batch_size=int(source.training_config["batch_size"]),
        device=device_info.device,
    )
    result = build_generalization_report(
        train=prepared.train,
        validation=prepared.validation,
        validation_predictions=validation_predictions,
        source_run_id=str(source.manifest["run_id"]),
        device=str(device_info.device),
        gpu_name=device_info.gpu_name,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03G_generalization_{timestamp}",
        result=result,
    )
