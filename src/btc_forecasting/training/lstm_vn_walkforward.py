from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import RobustScaler

from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm import (
    LOOKBACK_HOURS,
    LSTMRunResult,
    SequenceSamples,
    TrainingOutcome,
    _make_loader,
    _sample_identity,
    _write_json,
    build_sequence_samples,
    build_training_components,
    configure_determinism,
    fit_lstm,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    prepare_volatility_normalized_samples,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)
from btc_forecasting.training.lstm_vn_multiseed import (
    FROZEN_SEEDS,
    PRIMARY_METRIC_PATHS,
    _nested_metric,
    aggregate_seed_metrics,
    evaluate_seed,
)
from btc_forecasting.training.volatility_normalization import VOLATILITY_FEATURE_NAME

EXPERIMENT_ID = "E03-VN-WF4"
BLOCK_COUNT = 6
FOLD_COUNT = 4
INNER_VALIDATION_FRACTION = 0.15
ONE_HOUR = timedelta(hours=1)
TARGET_COLUMNS = ("bar_open_time", "decision_time", "target_time", TARGET_COLUMN)


@dataclass(frozen=True)
class TemporalBlock:
    number: int
    positions: np.ndarray


@dataclass(frozen=True)
class OuterFold:
    number: int
    outer_pool_positions: np.ndarray
    inner_train_positions: np.ndarray
    inner_validation_positions: np.ndarray
    outer_evaluation_positions: np.ndarray
    purged_outer_boundary_count: int
    purged_inner_boundary_count: int


@dataclass(frozen=True)
class FoldScopes:
    outer_pool: VolatilityNormalizedSamples
    inner_train: VolatilityNormalizedSamples
    inner_validation: VolatilityNormalizedSamples
    outer_evaluation: VolatilityNormalizedSamples


@dataclass(frozen=True)
class FixedEpochOutcome:
    model: LSTMRegressor
    history: list[dict[str, float | int]]
    duration_seconds: float


@dataclass(frozen=True)
class FoldSeedTraining:
    fold: int
    seed: int
    model: LSTMRegressor
    scaler: RobustScaler
    inner_history: list[dict[str, float | int]]
    refit_history: list[dict[str, float | int]]
    normalized_predictions: np.ndarray
    raw_predictions: np.ndarray
    metrics: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _load_frozen_train_targets(
    target_path: Path,
    *,
    split_metadata: dict[str, Any],
) -> pd.DataFrame:
    boundaries = split_metadata["split"]["boundaries"]
    train_start = _parse_utc(boundaries["train"]["decision_time_start_inclusive"])
    validation_start = _parse_utc(
        boundaries["validation"]["decision_time_start_inclusive"]
    )
    target_end = _parse_utc(boundaries["train"]["target_time_end_exclusive"])
    rows = pq.read_table(
        target_path,
        columns=list(TARGET_COLUMNS),
        filters=[
            ("decision_time", ">=", train_start),
            ("decision_time", "<", validation_start),
        ],
    ).to_pandas()
    decision = pd.to_datetime(rows["decision_time"], utc=True)
    target_time = pd.to_datetime(rows["target_time"], utc=True)
    retained = rows.loc[
        (decision >= train_start)
        & (decision < validation_start)
        & (target_time < target_end)
    ].sort_values("decision_time", kind="stable").reset_index(drop=True)
    expected = int(split_metadata["split"]["retained_rows"]["train"])
    if len(retained) != expected:
        raise ValueError(
            "Nested TRAIN target count does not match frozen metadata: "
            f"expected={expected}, actual={len(retained)}"
        )
    return retained


def load_frozen_train_vn_samples(*, project_root: Path) -> VolatilityNormalizedSamples:
    root = project_root.resolve()
    split_metadata = load_yaml(root / FROZEN_SPLIT_RELATIVE_PATH)
    targets = _load_frozen_train_targets(
        root / TARGET_RELATIVE_PATH,
        split_metadata=split_metadata,
    )
    maximum_anchor = pd.to_datetime(targets["bar_open_time"], utc=True).max()
    canonical = pq.read_table(
        root / CANONICAL_1H_RELATIVE_PATH,
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    ).to_pandas()
    features = compute_f0_features(canonical)
    return prepare_volatility_normalized_samples(
        build_sequence_samples(features, targets)
    )


def _purge_targets_reaching_boundary(
    positions: np.ndarray,
    decision_times: pd.DatetimeIndex,
    boundary: pd.Timestamp,
) -> np.ndarray:
    target_times = decision_times[positions] + ONE_HOUR
    return positions[target_times < boundary]


def build_temporal_design(
    decision_times: pd.DatetimeIndex,
) -> tuple[list[TemporalBlock], list[OuterFold]]:
    if len(decision_times) < BLOCK_COUNT:
        raise ValueError("WF4 requires six non-empty chronological TRAIN blocks")
    if decision_times.has_duplicates or not decision_times.is_monotonic_increasing:
        raise ValueError("WF4 decision times must be strictly ordered and unique")
    split_positions = np.array_split(np.arange(len(decision_times)), BLOCK_COUNT)
    blocks = [
        TemporalBlock(number=index, positions=positions)
        for index, positions in enumerate(split_positions, 1)
    ]
    folds: list[OuterFold] = []
    for fold_number in range(1, FOLD_COUNT + 1):
        outer_positions = blocks[fold_number + 1].positions
        outer_start = decision_times[outer_positions[0]]
        unpurged_pool = np.concatenate(
            [block.positions for block in blocks[: fold_number + 1]]
        )
        outer_pool = _purge_targets_reaching_boundary(
            unpurged_pool,
            decision_times,
            outer_start,
        )
        purged_outer = len(unpurged_pool) - len(outer_pool)
        inner_validation_count = max(
            1,
            math.floor(len(outer_pool) * INNER_VALIDATION_FRACTION),
        )
        if inner_validation_count >= len(outer_pool):
            raise ValueError("WF4 outer pool is too small for nested inner splitting")
        inner_validation = outer_pool[-inner_validation_count:]
        inner_start = decision_times[inner_validation[0]]
        unpurged_inner_train = outer_pool[:-inner_validation_count]
        inner_train = _purge_targets_reaching_boundary(
            unpurged_inner_train,
            decision_times,
            inner_start,
        )
        purged_inner = len(unpurged_inner_train) - len(inner_train)
        folds.append(
            OuterFold(
                number=fold_number,
                outer_pool_positions=outer_pool,
                inner_train_positions=inner_train,
                inner_validation_positions=inner_validation,
                outer_evaluation_positions=outer_positions,
                purged_outer_boundary_count=purged_outer,
                purged_inner_boundary_count=purged_inner,
            )
        )
    return blocks, folds


def _subset_sequence(samples: SequenceSamples, positions: np.ndarray) -> SequenceSamples:
    return SequenceSamples(
        features=samples.features[positions],
        targets=samples.targets[positions],
        decision_times=samples.decision_times[positions],
        candidate_count=len(positions),
        excluded_lookback_count=0,
    )


def subset_vn_samples(
    samples: VolatilityNormalizedSamples,
    positions: np.ndarray,
) -> VolatilityNormalizedSamples:
    return VolatilityNormalizedSamples(
        raw=_subset_sequence(samples.raw, positions),
        normalized=_subset_sequence(samples.normalized, positions),
        sigma=samples.sigma[positions],
        exclusion_count=0,
        original_eligible_count=len(positions),
    )


def prepare_fold_scopes(
    samples: VolatilityNormalizedSamples,
    fold: OuterFold,
) -> FoldScopes:
    return FoldScopes(
        outer_pool=subset_vn_samples(samples, fold.outer_pool_positions),
        inner_train=subset_vn_samples(samples, fold.inner_train_positions),
        inner_validation=subset_vn_samples(
            samples,
            fold.inner_validation_positions,
        ),
        outer_evaluation=subset_vn_samples(
            samples,
            fold.outer_evaluation_positions,
        ),
    )


def fit_scope_scaler(train: SequenceSamples) -> tuple[RobustScaler, SequenceSamples]:
    if train.features.ndim != 3 or train.features.shape[2] != len(F0_FEATURE_NAMES):
        raise ValueError("WF4 scaler requires frozen F0 sequences")
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
    )
    shape = train.features.shape
    scaled = scaler.fit_transform(
        train.features.reshape(-1, len(F0_FEATURE_NAMES))
    ).reshape(shape)
    return scaler, _replace_features(train, scaled)


def transform_scope(
    samples: SequenceSamples,
    scaler: RobustScaler,
) -> SequenceSamples:
    shape = samples.features.shape
    transformed = scaler.transform(
        samples.features.reshape(-1, len(F0_FEATURE_NAMES))
    ).reshape(shape)
    return _replace_features(samples, transformed)


def _replace_features(
    samples: SequenceSamples,
    features: np.ndarray,
) -> SequenceSamples:
    return SequenceSamples(
        features=features.astype(np.float32, copy=False),
        targets=samples.targets,
        decision_times=samples.decision_times,
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def selected_refit_epoch_count(
    best_epoch: int,
    *,
    max_epochs: int,
) -> int:
    if best_epoch < 1 or best_epoch > max_epochs:
        raise ValueError("Selected best_epoch must be inside the frozen epoch budget")
    return best_epoch


def fit_lstm_fixed_epochs(
    model: LSTMRegressor,
    *,
    train: SequenceSamples,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
) -> FixedEpochOutcome:
    epoch_count = selected_refit_epoch_count(
        epochs,
        max_epochs=int(training_config["max_epochs"]),
    )
    loader = _make_loader(
        train,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model.to(device)
    optimizer, loss_function, scheduler = build_training_components(
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
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(batch_targets)
            count += len(batch_targets)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_mse_loss": loss_sum / count,
            }
        )
        scheduler.step()
    return FixedEpochOutcome(
        model=model,
        history=history,
        duration_seconds=time.perf_counter() - started,
    )


def _new_model(model_config: dict[str, Any]) -> LSTMRegressor:
    return LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        configured_dropout=float(model_config["dropout"]),
    )


def train_fold_seed(
    *,
    fold: int,
    seed: int,
    scopes: FoldScopes,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> FoldSeedTraining:
    configure_determinism(seed)
    inner_scaler, scaled_inner_train = fit_scope_scaler(
        scopes.inner_train.normalized
    )
    scaled_inner_validation = transform_scope(
        scopes.inner_validation.normalized,
        inner_scaler,
    )
    inner_outcome: TrainingOutcome = fit_lstm(
        _new_model(model_config),
        train=scaled_inner_train,
        validation=scaled_inner_validation,
        training_config=training_config,
        seed=seed,
        device=device,
    )
    refit_epochs = selected_refit_epoch_count(
        inner_outcome.best_epoch,
        max_epochs=int(training_config["max_epochs"]),
    )

    configure_determinism(seed)
    outer_scaler, scaled_outer_pool = fit_scope_scaler(scopes.outer_pool.normalized)
    refit_outcome = fit_lstm_fixed_epochs(
        _new_model(model_config),
        train=scaled_outer_pool,
        training_config=training_config,
        seed=seed,
        device=device,
        epochs=refit_epochs,
    )
    scaled_outer_evaluation = transform_scope(
        scopes.outer_evaluation.normalized,
        outer_scaler,
    )
    normalized_predictions = predict_lstm(
        refit_outcome.model,
        scaled_outer_evaluation.features,
        batch_size=int(training_config["batch_size"]),
        device=device,
    )
    metrics = evaluate_seed(
        seed=seed,
        train=scopes.outer_pool,
        validation=scopes.outer_evaluation,
        normalized_predictions=normalized_predictions,
        best_epoch=inner_outcome.best_epoch,
        epochs_trained=inner_outcome.epochs_trained,
        duration_seconds=(
            inner_outcome.duration_seconds + refit_outcome.duration_seconds
        ),
    )
    metrics["training"] = {
        "inner_best_epoch": inner_outcome.best_epoch,
        "inner_epochs_trained": inner_outcome.epochs_trained,
        "inner_duration_seconds": inner_outcome.duration_seconds,
        "refit_epoch_count": len(refit_outcome.history),
        "refit_duration_seconds": refit_outcome.duration_seconds,
    }
    return FoldSeedTraining(
        fold=fold,
        seed=seed,
        model=refit_outcome.model,
        scaler=outer_scaler,
        inner_history=inner_outcome.history,
        refit_history=refit_outcome.history,
        normalized_predictions=normalized_predictions,
        raw_predictions=reconstruct_raw_predictions(
            normalized_predictions,
            scopes.outer_evaluation.sigma,
        ),
        metrics=metrics,
    )


def aggregate_evaluations(
    evaluations: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    if not evaluations:
        raise ValueError("WF4 aggregation requires evaluations")
    result: dict[str, dict[str, float]] = {}
    for name, path in PRIMARY_METRIC_PATHS.items():
        values = np.asarray(
            [_nested_metric(evaluation, path) for evaluation in evaluations],
            dtype=np.float64,
        )
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return result


def temporal_stability_gate(
    fold_aggregates: list[dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> tuple[list[bool], bool]:
    if len(fold_aggregates) != FOLD_COUNT:
        raise ValueError("WF4 stability gate requires exactly four fold aggregates")
    positive = [
        (
            aggregate["mae_skill"]["mean"] > 0.0
            and aggregate["rmse_skill"]["mean"] > 0.0
            and aggregate["r2"]["mean"] > 0.0
            and aggregate["pearson_ic"]["mean"] > 0.0
            and aggregate["spearman_rank_ic"]["mean"] > 0.0
            and aggregate["directional_accuracy"]["mean"] > 0.50
        )
        for aggregate in fold_aggregates
    ]
    stable = (
        sum(positive) >= 3
        and overall["mae_skill"]["mean"] > 0.0
        and overall["rmse_skill"]["mean"] > 0.0
        and overall["r2"]["mean"] > 0.0
    )
    return positive, stable


def _scope_summary(
    samples: VolatilityNormalizedSamples,
) -> dict[str, object]:
    decisions = samples.normalized.decision_times
    return {
        "start_decision_time": decisions[0].isoformat(),
        "end_decision_time": decisions[-1].isoformat(),
        "sample_count": len(decisions),
        "sample_identity_sha256": _sample_identity(decisions),
    }


def temporal_design_report(
    samples: VolatilityNormalizedSamples,
    blocks: list[TemporalBlock],
    folds: list[OuterFold],
) -> dict[str, object]:
    block_report = []
    for block in blocks:
        subset = subset_vn_samples(samples, block.positions)
        block_report.append({"block": f"B{block.number}", **_scope_summary(subset)})
    fold_report = []
    for fold in folds:
        scopes = prepare_fold_scopes(samples, fold)
        fold_report.append(
            {
                "fold": fold.number,
                "train_blocks": [f"B{number}" for number in range(1, fold.number + 2)],
                "outer_evaluation_block": f"B{fold.number + 2}",
                "outer_pool": _scope_summary(scopes.outer_pool),
                "inner_train": _scope_summary(scopes.inner_train),
                "inner_validation": _scope_summary(scopes.inner_validation),
                "outer_evaluation": _scope_summary(scopes.outer_evaluation),
                "purged_outer_boundary_count": fold.purged_outer_boundary_count,
                "purged_inner_boundary_count": fold.purged_inner_boundary_count,
            }
        )
    return {
        "source_scope": "frozen TRAIN only",
        "block_construction": "six contiguous approximately equal-count blocks; target values unused",
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "inner_validation_rounding": "floor",
        "blocks": block_report,
        "folds": fold_report,
        "original_validation": "NOT READ OR USED",
        "test": "NOT READ OR USED",
    }


def run_lstm_vn_walkforward_experiment(*, project_root: Path) -> LSTMRunResult:
    root = project_root.resolve()
    resolved = resolve_vn_mse_configuration(project_root=root)
    device_info = require_official_cuda()
    samples = load_frozen_train_vn_samples(project_root=root)
    blocks, folds = build_temporal_design(samples.normalized.decision_times)
    design = temporal_design_report(samples, blocks, folds)

    fold_results: list[dict[str, object]] = []
    trained: list[FoldSeedTraining] = []
    all_evaluations: list[dict[str, object]] = []
    for fold in folds:
        scopes = prepare_fold_scopes(samples, fold)
        seed_trainings = [
            train_fold_seed(
                fold=fold.number,
                seed=seed,
                scopes=scopes,
                model_config=resolved.model,
                training_config=resolved.training,
                device=device_info.device,
            )
            for seed in FROZEN_SEEDS
        ]
        trained.extend(seed_trainings)
        seed_metrics = [member.metrics for member in seed_trainings]
        aggregate = aggregate_seed_metrics(seed_metrics)
        all_evaluations.extend(seed_metrics)
        fold_results.append(
            {
                "fold": fold.number,
                "outer_evaluation": _scope_summary(scopes.outer_evaluation),
                "outer_target_statistics": {
                    "sample_count": len(scopes.outer_evaluation.raw.targets),
                    "raw_target_std": float(
                        np.std(scopes.outer_evaluation.raw.targets, ddof=0)
                    ),
                    "normalized_target_std": float(
                        np.std(scopes.outer_evaluation.normalized.targets, ddof=0)
                    ),
                },
                "per_seed": seed_metrics,
                "five_seed_aggregate": aggregate,
            }
        )
    overall = aggregate_evaluations(all_evaluations)
    fold_aggregates = [
        fold_result["five_seed_aggregate"] for fold_result in fold_results
    ]
    positive, stable = temporal_stability_gate(fold_aggregates, overall)  # type: ignore[arg-type]
    for fold_result, is_positive in zip(fold_results, positive, strict=True):
        fold_result["positive_fold"] = is_positive
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does the seed-stable F0 VN-LSTM edge persist across unseen TRAIN periods?",
        "experiment_type": "nested_train_only_temporal_robustness",
        "frozen_seeds": list(FROZEN_SEEDS),
        "temporal_design": design,
        "fold_results": fold_results,
        "overall_20_evaluation_aggregate": overall,
        "stability_gate": {
            "positive_fold_count": sum(positive),
            "required_positive_fold_count": 3,
            "fold_positive_flags": positive,
            "PRELIMINARY_TEMPORAL_STABLE": stable,
            "interpretation": (
                "Temporal robustness only; not economic profitability or final thesis validity."
            ),
        },
        "evaluated_scope": "frozen TRAIN outer blocks only",
        "original_validation": "NOT READ OR USED",
        "test_set": "NOT READ OR USED",
        "training": {
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "model_config": resolved.model,
            "training_config": resolved.training,
        },
    }

    run_id = f"{EXPERIMENT_ID}_1h_F0_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.parquet"
    history_path = run_directory / "training_history.json"
    resolved_config_path = run_directory / "resolved_config.json"
    manifest = create_run_manifest(
        project_root=root,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
    )
    manifest.update(
        {
            "base_experiment_id": "E03-VN-MS5",
            "experiment_type": "nested_train_only_temporal_robustness",
            "frozen_seeds": list(FROZEN_SEEDS),
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "model_config": resolved.model,
            "training_config": resolved.training,
            "feature_set": "F0",
            "feature_names": list(F0_FEATURE_NAMES),
            "temporal_design": design,
            "evaluated_scope": "frozen TRAIN outer blocks only",
            "original_validation": "NOT READ OR USED",
            "test_set": "NOT READ OR USED",
            "data": {
                "canonical_1h": {
                    "path": CANONICAL_1H_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / CANONICAL_1H_RELATIVE_PATH),
                },
                "target": {
                    "path": TARGET_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / TARGET_RELATIVE_PATH),
                },
            },
            "split": {
                "path": FROZEN_SPLIT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(root / FROZEN_SPLIT_RELATIVE_PATH),
            },
            "artifacts": {
                "metrics": metrics_path.relative_to(root).as_posix(),
                "predictions": predictions_path.relative_to(root).as_posix(),
                "training_history": history_path.relative_to(root).as_posix(),
                "resolved_config": resolved_config_path.relative_to(root).as_posix(),
                "model_directory": model_directory.relative_to(root).as_posix(),
                "scaler_directory": scaler_directory.relative_to(root).as_posix(),
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    _write_json(metrics_path, result)
    _write_json(
        history_path,
        {
            f"fold_{member.fold}_seed_{member.seed}": {
                "inner_selection": member.inner_history,
                "outer_refit": member.refit_history,
            }
            for member in trained
        },
    )
    _write_json(
        resolved_config_path,
        {
            "experiment": {
                "id": EXPERIMENT_ID,
                "base_experiment": "E03-VN-MS5",
                "type": "nested_train_only_temporal_robustness",
                "frozen_seeds": list(FROZEN_SEEDS),
            },
            "temporal_design": design,
            "feature_set": resolved.feature,
            "model": resolved.model,
            "training": resolved.training,
        },
    )

    prediction_rows: list[pd.DataFrame] = []
    for member in trained:
        scopes = prepare_fold_scopes(samples, folds[member.fold - 1])
        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": member.fold,
                    "seed": member.seed,
                    "decision_time": scopes.outer_evaluation.raw.decision_times,
                    TARGET_COLUMN: scopes.outer_evaluation.raw.targets,
                    VOLATILITY_FEATURE_NAME: scopes.outer_evaluation.sigma,
                    "normalized_target": scopes.outer_evaluation.normalized.targets,
                    "normalized_prediction": member.normalized_predictions,
                    "raw_return_prediction": member.raw_predictions,
                }
            )
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    predictions.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)

    model_directory.mkdir(parents=True, exist_ok=False)
    scaler_directory.mkdir(parents=True, exist_ok=False)
    for member in trained:
        stem = f"fold_{member.fold}_seed_{member.seed}"
        checkpoint = model_directory / f"{stem}_lstm.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": member.model.state_dict(),
                "experiment_id": EXPERIMENT_ID,
                "fold": member.fold,
                "seed": member.seed,
                "selected_best_epoch": member.metrics["training"]["inner_best_epoch"],  # type: ignore[index]
                "refit_epoch_count": member.metrics["training"]["refit_epoch_count"],  # type: ignore[index]
                "model_config": resolved.model,
                "training_config": resolved.training,
                "feature_names": list(F0_FEATURE_NAMES),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        joblib.dump(member.scaler, scaler_directory / f"{stem}_scaler.joblib")
    return LSTMRunResult(
        metrics_path=metrics_path,
        run_directory=run_directory,
        result=result,
    )
