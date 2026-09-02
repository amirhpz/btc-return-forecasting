from __future__ import annotations

import json
import math
import os
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

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.horizon_audit import (
    AUDIT_OUTPUT_RELATIVE_PATH,
    FROZEN_HORIZONS_HOURS,
    HorizonConstruction,
    _sample_identity,
    common_anchor_times,
    construct_horizon_targets,
    f0_eligible_anchor_data,
)
from btc_forecasting.training.lstm import (
    LSTMRunResult,
    SequenceSamples,
    TrainingOutcome,
    _write_json,
    configure_determinism,
    fit_lstm,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_generalization import _correlation
from btc_forecasting.training.lstm_vn_mse import resolve_vn_mse_configuration
from btc_forecasting.training.lstm_vn_multiseed import (
    FROZEN_SEEDS,
    aggregate_seed_metrics,
)
from btc_forecasting.training.lstm_vn_walkforward import (
    BLOCK_COUNT,
    FOLD_COUNT,
    INNER_VALIDATION_FRACTION,
    FixedEpochOutcome,
    OuterFold,
    TemporalBlock,
    aggregate_evaluations,
    fit_lstm_fixed_epochs,
    fit_scope_scaler,
    selected_refit_epoch_count,
    temporal_stability_gate,
    transform_scope,
)
from btc_forecasting.training.volatility_normalization import VOLATILITY_FEATURE_NAME

EXPERIMENT_ID = "E04-H-WF4"
EXPECTED_COMMON_COUNT = 53_013
EXPECTED_COMMON_IDENTITY = (
    "84a670ec07c61459641282970697bed57682aaccfade75c3684bf38ccd771ac1"
)
MAXIMUM_HORIZON_HOURS = 12
ONE_HOUR = timedelta(hours=1)
LOOKBACK_HOURS = 24


@dataclass(frozen=True)
class PairedHorizonSamples:
    features: np.ndarray
    decision_times: pd.DatetimeIndex
    sigma: np.ndarray
    raw_targets: dict[int, np.ndarray]
    normalized_targets: dict[int, np.ndarray]


@dataclass(frozen=True)
class HorizonFoldTraining:
    horizon_hours: int
    fold: int
    seed: int
    model: LSTMRegressor
    scaler: RobustScaler
    inner_history: list[dict[str, float | int]]
    refit_history: list[dict[str, float | int]]
    normalized_predictions: np.ndarray
    raw_predictions: np.ndarray
    metrics: dict[str, object]


def validate_horizon_audit(project_root: Path) -> dict[str, object]:
    path = project_root.resolve() / AUDIT_OUTPUT_RELATIVE_PATH
    with path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    common = audit.get("common_paired_anchors")
    if not isinstance(common, dict):
        raise ValueError("E04-H-A audit has no common paired-anchor record")
    if common.get("sample_count") != EXPECTED_COMMON_COUNT:
        raise ValueError("E04-H-A COMMON anchor count does not match the frozen gate")
    if common.get("sample_identity_sha256") != EXPECTED_COMMON_IDENTITY:
        raise ValueError("E04-H-A COMMON anchor identity does not match the frozen gate")
    if tuple(audit.get("frozen_horizons_hours", ())) != FROZEN_HORIZONS_HOURS:
        raise ValueError("E04-H-A audit horizons do not match the frozen study")
    if audit.get("scope", {}).get("original_validation") != "NOT READ OR USED":
        raise ValueError("E04-H-A audit did not preserve original Validation closure")
    if audit.get("scope", {}).get("test") != "NOT READ OR USED":
        raise ValueError("E04-H-A audit did not preserve TEST closure")
    return audit


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _targets_on_common(
    construction: HorizonConstruction,
    common: pd.DatetimeIndex,
) -> np.ndarray:
    frame = construction.valid.set_index("decision_time")
    values = frame.reindex(common)["future_log_return"].to_numpy(np.float64)
    if len(values) != len(common) or not np.isfinite(values).all():
        raise ValueError("Horizon targets do not cover the exact COMMON anchor set")
    return values


def load_paired_horizon_samples(*, project_root: Path) -> PairedHorizonSamples:
    root = project_root.resolve()
    validate_horizon_audit(root)
    split = load_yaml(root / FROZEN_SPLIT_RELATIVE_PATH)
    boundaries = split["split"]["boundaries"]
    train_start = _parse_utc(boundaries["train"]["decision_time_start_inclusive"])
    validation_start = _parse_utc(
        boundaries["validation"]["decision_time_start_inclusive"]
    )
    target_end = _parse_utc(boundaries["train"]["target_time_end_exclusive"])
    latest_allowed_open = target_end - ONE_HOUR
    canonical = pq.read_table(
        root / CANONICAL_1H_RELATIVE_PATH,
        filters=[("open_time", "<", latest_allowed_open)],
    ).to_pandas()
    canonical_times = pd.DatetimeIndex(pd.to_datetime(canonical["open_time"], utc=True))
    decisions = canonical_times + ONE_HOUR
    anchor_times = canonical_times[
        (decisions >= train_start) & (decisions < validation_start)
    ]
    constructions = {
        horizon: construct_horizon_targets(
            canonical,
            anchor_times,
            horizon_hours=horizon,
            target_scope_end_exclusive=target_end,
        )
        for horizon in FROZEN_HORIZONS_HOURS
    }
    f0 = compute_f0_features(canonical)
    eligible = f0_eligible_anchor_data(f0, anchor_times)
    eligible_decisions = pd.DatetimeIndex(
        pd.to_datetime(eligible["decision_time"], utc=True)
    )
    common = common_anchor_times(
        eligible_decisions,
        {
            horizon: pd.DatetimeIndex(
                pd.to_datetime(construction.valid["decision_time"], utc=True)
            )
            for horizon, construction in constructions.items()
        },
    )
    if len(common) != EXPECTED_COMMON_COUNT or _sample_identity(common) != EXPECTED_COMMON_IDENTITY:
        raise ValueError("Reconstructed COMMON anchors do not match E04-H-A")

    feature_times = pd.DatetimeIndex(pd.to_datetime(f0["open_time"], utc=True))
    feature_values = f0.loc[:, list(F0_FEATURE_NAMES)].to_numpy(np.float64)
    endpoint_positions = feature_times.get_indexer(common - ONE_HOUR)
    windows = np.empty(
        (len(common), LOOKBACK_HOURS, len(F0_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for row, end in enumerate(endpoint_positions):
        start = end - LOOKBACK_HOURS + 1
        window = feature_values[start : end + 1]
        if end < 0 or start < 0 or window.shape != windows.shape[1:]:
            raise ValueError("COMMON anchor lacks its frozen 24-hour F0 window")
        if feature_times[end] - feature_times[start] != 23 * ONE_HOUR:
            raise ValueError("COMMON F0 window crosses a missing real hour")
        windows[row] = window
    if not np.isfinite(windows).all():
        raise ValueError("COMMON F0 windows must be finite")
    sigma = windows[:, -1, F0_FEATURE_NAMES.index(VOLATILITY_FEATURE_NAME)].astype(
        np.float64
    )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("COMMON anchors require finite positive sigma_24h")
    raw = {
        horizon: _targets_on_common(construction, common)
        for horizon, construction in constructions.items()
    }
    normalized = {horizon: raw[horizon] / sigma for horizon in FROZEN_HORIZONS_HOURS}
    if any(not np.isfinite(values).all() for values in normalized.values()):
        raise ValueError("Volatility-normalized horizon targets must be finite")
    return PairedHorizonSamples(windows, common, sigma, raw, normalized)


def _purge_for_maximum_horizon(
    positions: np.ndarray,
    decision_times: pd.DatetimeIndex,
    boundary: pd.Timestamp,
) -> np.ndarray:
    target_times = decision_times[positions] + MAXIMUM_HORIZON_HOURS * ONE_HOUR
    return positions[target_times < boundary]


def build_paired_temporal_design(
    decision_times: pd.DatetimeIndex,
) -> tuple[list[TemporalBlock], list[OuterFold]]:
    if len(decision_times) < BLOCK_COUNT:
        raise ValueError("E04-H-WF4 requires six non-empty chronological blocks")
    if decision_times.has_duplicates or not decision_times.is_monotonic_increasing:
        raise ValueError("COMMON decision times must be strictly ordered and unique")
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
        unpurged_outer_pool = np.concatenate(
            [block.positions for block in blocks[: fold_number + 1]]
        )
        outer_pool = _purge_for_maximum_horizon(
            unpurged_outer_pool,
            decision_times,
            outer_start,
        )
        inner_count = max(1, math.floor(len(outer_pool) * INNER_VALIDATION_FRACTION))
        if inner_count >= len(outer_pool):
            raise ValueError("Outer pool is too small for nested inner validation")
        inner_validation = outer_pool[-inner_count:]
        inner_start = decision_times[inner_validation[0]]
        unpurged_inner_train = outer_pool[:-inner_count]
        inner_train = _purge_for_maximum_horizon(
            unpurged_inner_train,
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
                purged_outer_boundary_count=len(unpurged_outer_pool) - len(outer_pool),
                purged_inner_boundary_count=len(unpurged_inner_train) - len(inner_train),
            )
        )
    return blocks, folds


def relative_horizon_scales(
    normalized_targets: dict[int, np.ndarray],
    training_positions: np.ndarray,
) -> dict[int, float]:
    if tuple(sorted(normalized_targets)) != FROZEN_HORIZONS_HOURS:
        raise ValueError("Relative scaling requires exactly the four frozen horizons")
    if len(training_positions) == 0:
        raise ValueError("Relative scaling requires a non-empty training scope")
    rms = {
        horizon: float(
            np.sqrt(np.mean(np.square(normalized_targets[horizon][training_positions])))
        )
        for horizon in FROZEN_HORIZONS_HOURS
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in rms.values()):
        raise ValueError("Relative horizon RMS values must be finite and positive")
    return {
        horizon: 1.0 if horizon == 1 else rms[horizon] / rms[1]
        for horizon in FROZEN_HORIZONS_HOURS
    }


def _sequence(
    samples: PairedHorizonSamples,
    positions: np.ndarray,
    *,
    horizon_hours: int,
    scale: float,
) -> SequenceSamples:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Horizon target scale must be finite and positive")
    return SequenceSamples(
        features=samples.features[positions],
        targets=samples.normalized_targets[horizon_hours][positions] / scale,
        decision_times=samples.decision_times[positions],
        candidate_count=len(positions),
        excluded_lookback_count=0,
    )


def _new_model(model_config: dict[str, Any]) -> LSTMRegressor:
    return LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        configured_dropout=float(model_config["dropout"]),
    )


def evaluate_horizon_seed(
    *,
    horizon_hours: int,
    fold: int,
    seed: int,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    samples: PairedHorizonSamples,
    training_target_predictions: np.ndarray,
    outer_scale: float,
    training_metadata: dict[str, float | int],
) -> tuple[dict[str, object], np.ndarray]:
    z_predictions = np.asarray(training_target_predictions, dtype=np.float64)
    q_predictions = z_predictions * outer_scale
    raw_predictions = q_predictions * samples.sigma[evaluation_positions]
    raw_targets = samples.raw_targets[horizon_hours][evaluation_positions]
    z_targets = (
        samples.normalized_targets[horizon_hours][evaluation_positions] / outer_scale
    )
    raw_metrics = regression_metrics(raw_targets, raw_predictions)
    zero_metrics = regression_loss_metrics(
        raw_targets,
        zero_return_prediction(len(raw_targets)),
    )
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    metrics: dict[str, object] = {
        "horizon_hours": horizon_hours,
        "fold": fold,
        "seed": seed,
        "train_sample_count": int(len(train_positions)),
        "validation_sample_count": int(len(evaluation_positions)),
        "train_sample_identity_sha256": _sample_identity(
            samples.decision_times[train_positions]
        ),
        "validation_sample_identity_sha256": _sample_identity(
            samples.decision_times[evaluation_positions]
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
            "mse": float(np.mean(np.square(z_predictions - z_targets))),
            "pearson_ic": _correlation(z_targets, z_predictions, rank=False),
            "spearman_rank_ic": _correlation(z_targets, z_predictions, rank=True),
            "prediction_std": float(np.std(z_predictions, ddof=0)),
            "target_std": float(np.std(z_targets, ddof=0)),
        },
        "training": training_metadata,
        "evaluated_scope": "frozen TRAIN outer block only",
        "original_validation": "NOT READ OR USED",
        "test_set": "NOT READ OR USED",
    }
    return metrics, raw_predictions


def train_horizon_fold_seed(
    *,
    horizon_hours: int,
    fold: OuterFold,
    seed: int,
    samples: PairedHorizonSamples,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> HorizonFoldTraining:
    inner_scales = relative_horizon_scales(
        samples.normalized_targets,
        fold.inner_train_positions,
    )
    inner_scale = inner_scales[horizon_hours]
    configure_determinism(seed)
    inner_train = _sequence(
        samples,
        fold.inner_train_positions,
        horizon_hours=horizon_hours,
        scale=inner_scale,
    )
    inner_validation = _sequence(
        samples,
        fold.inner_validation_positions,
        horizon_hours=horizon_hours,
        scale=inner_scale,
    )
    inner_scaler, scaled_inner_train = fit_scope_scaler(inner_train)
    scaled_inner_validation = transform_scope(inner_validation, inner_scaler)
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

    outer_scales = relative_horizon_scales(
        samples.normalized_targets,
        fold.outer_pool_positions,
    )
    outer_scale = outer_scales[horizon_hours]
    configure_determinism(seed)
    outer_pool = _sequence(
        samples,
        fold.outer_pool_positions,
        horizon_hours=horizon_hours,
        scale=outer_scale,
    )
    outer_evaluation = _sequence(
        samples,
        fold.outer_evaluation_positions,
        horizon_hours=horizon_hours,
        scale=outer_scale,
    )
    outer_scaler, scaled_outer_pool = fit_scope_scaler(outer_pool)
    refit_outcome: FixedEpochOutcome = fit_lstm_fixed_epochs(
        _new_model(model_config),
        train=scaled_outer_pool,
        training_config=training_config,
        seed=seed,
        device=device,
        epochs=refit_epochs,
    )
    scaled_outer_evaluation = transform_scope(outer_evaluation, outer_scaler)
    z_predictions = predict_lstm(
        refit_outcome.model,
        scaled_outer_evaluation.features,
        batch_size=int(training_config["batch_size"]),
        device=device,
    )
    training_metadata: dict[str, float | int] = {
        "inner_best_epoch": inner_outcome.best_epoch,
        "inner_epochs_trained": inner_outcome.epochs_trained,
        "inner_validation_loss": inner_outcome.best_validation_loss,
        "inner_c_h": inner_scale,
        "outer_c_h": outer_scale,
        "refit_epoch_count": len(refit_outcome.history),
        "inner_duration_seconds": inner_outcome.duration_seconds,
        "refit_duration_seconds": refit_outcome.duration_seconds,
        "total_duration_seconds": (
            inner_outcome.duration_seconds + refit_outcome.duration_seconds
        ),
    }
    metrics, raw_predictions = evaluate_horizon_seed(
        horizon_hours=horizon_hours,
        fold=fold.number,
        seed=seed,
        train_positions=fold.outer_pool_positions,
        evaluation_positions=fold.outer_evaluation_positions,
        samples=samples,
        training_target_predictions=z_predictions,
        outer_scale=outer_scale,
        training_metadata=training_metadata,
    )
    return HorizonFoldTraining(
        horizon_hours=horizon_hours,
        fold=fold.number,
        seed=seed,
        model=refit_outcome.model,
        scaler=outer_scaler,
        inner_history=inner_outcome.history,
        refit_history=refit_outcome.history,
        normalized_predictions=z_predictions,
        raw_predictions=raw_predictions,
        metrics=metrics,
    )


def _scope_summary(times: pd.DatetimeIndex, positions: np.ndarray) -> dict[str, object]:
    selected = times[positions]
    return {
        "start_decision_time": selected[0].isoformat(),
        "end_decision_time": selected[-1].isoformat(),
        "sample_count": len(selected),
        "sample_identity_sha256": _sample_identity(selected),
    }


def paired_temporal_design_report(
    samples: PairedHorizonSamples,
    blocks: list[TemporalBlock],
    folds: list[OuterFold],
) -> dict[str, object]:
    return {
        "common_anchor_count": len(samples.decision_times),
        "common_anchor_identity_sha256": _sample_identity(samples.decision_times),
        "block_construction": "six contiguous approximately equal-count COMMON-anchor blocks; targets unused",
        "maximum_horizon_boundary_purge_hours": MAXIMUM_HORIZON_HOURS,
        "blocks": [
            {"block": f"B{block.number}", **_scope_summary(samples.decision_times, block.positions)}
            for block in blocks
        ],
        "folds": [
            {
                "fold": fold.number,
                "train_blocks": [f"B{number}" for number in range(1, fold.number + 2)],
                "outer_evaluation_block": f"B{fold.number + 2}",
                "outer_pool": _scope_summary(samples.decision_times, fold.outer_pool_positions),
                "inner_train": _scope_summary(samples.decision_times, fold.inner_train_positions),
                "inner_validation": _scope_summary(
                    samples.decision_times,
                    fold.inner_validation_positions,
                ),
                "outer_evaluation": _scope_summary(
                    samples.decision_times,
                    fold.outer_evaluation_positions,
                ),
                "purged_outer_boundary_count": fold.purged_outer_boundary_count,
                "purged_inner_boundary_count": fold.purged_inner_boundary_count,
            }
            for fold in folds
        ],
        "paired_across_all_horizons": True,
        "original_validation": "NOT READ OR USED",
        "test": "NOT READ OR USED",
    }


def select_regression_horizon(
    horizon_results: dict[int, dict[str, object]],
) -> int | None:
    passing = [
        horizon
        for horizon in FROZEN_HORIZONS_HOURS
        if bool(horizon_results[horizon]["HORIZON_REGRESSION_STABLE"])
    ]
    if not passing:
        return None

    def rank(horizon: int) -> tuple[float, float, float, float]:
        result = horizon_results[horizon]
        overall = result["overall_20_evaluation_aggregate"]
        if not isinstance(overall, dict):
            raise TypeError("Horizon aggregate must be a mapping")
        return (
            float(result["regression_positive_fold_count"]),
            float(overall["rmse_skill"]["mean"]),
            float(overall["mae_skill"]["mean"]),
            float(overall["r2"]["mean"]),
        )

    ordered = sorted(passing, key=rank, reverse=True)
    if len(ordered) > 1 and rank(ordered[0]) == rank(ordered[1]):
        raise ValueError("Frozen horizon selection criteria produce an unresolved tie")
    return ordered[0]


def _direction_rank_diagnostics(
    fold_aggregates: list[dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> dict[str, float | int]:
    return {
        "folds_with_mean_directional_accuracy_gt_0_50": sum(
            fold["directional_accuracy"]["mean"] > 0.50 for fold in fold_aggregates
        ),
        "folds_with_mean_pearson_ic_gt_0": sum(
            fold["pearson_ic"]["mean"] > 0.0 for fold in fold_aggregates
        ),
        "folds_with_mean_spearman_rank_ic_gt_0": sum(
            fold["spearman_rank_ic"]["mean"] > 0.0 for fold in fold_aggregates
        ),
        "overall_mean_directional_accuracy": overall["directional_accuracy"]["mean"],
        "overall_mean_pearson_ic": overall["pearson_ic"]["mean"],
        "overall_mean_spearman_rank_ic": overall["spearman_rank_ic"]["mean"],
    }


def run_lstm_horizon_walkforward_experiment(*, project_root: Path) -> LSTMRunResult:
    root = project_root.resolve()
    audit = validate_horizon_audit(root)
    resolved = resolve_vn_mse_configuration(project_root=root)
    device_info = require_official_cuda()
    samples = load_paired_horizon_samples(project_root=root)
    blocks, folds = build_paired_temporal_design(samples.decision_times)
    design = paired_temporal_design_report(samples, blocks, folds)

    trained: list[HorizonFoldTraining] = []
    horizon_results: dict[int, dict[str, object]] = {}
    for horizon in FROZEN_HORIZONS_HOURS:
        fold_results: list[dict[str, object]] = []
        all_evaluations: list[dict[str, object]] = []
        for fold in folds:
            members = [
                train_horizon_fold_seed(
                    horizon_hours=horizon,
                    fold=fold,
                    seed=seed,
                    samples=samples,
                    model_config=resolved.model,
                    training_config=resolved.training,
                    device=device_info.device,
                )
                for seed in FROZEN_SEEDS
            ]
            trained.extend(members)
            per_seed = [member.metrics for member in members]
            aggregate = aggregate_seed_metrics(per_seed)
            all_evaluations.extend(per_seed)
            fold_results.append(
                {
                    "fold": fold.number,
                    "outer_evaluation": _scope_summary(
                        samples.decision_times,
                        fold.outer_evaluation_positions,
                    ),
                    "per_seed": per_seed,
                    "five_seed_aggregate": aggregate,
                }
            )
        overall = aggregate_evaluations(all_evaluations)
        fold_aggregates = [result["five_seed_aggregate"] for result in fold_results]
        positive, stable = temporal_stability_gate(fold_aggregates, overall)  # type: ignore[arg-type]
        for result, flag in zip(fold_results, positive, strict=True):
            result["REGRESSION_POSITIVE"] = flag
        horizon_results[horizon] = {
            "horizon_hours": horizon,
            "fold_results": fold_results,
            "overall_20_evaluation_aggregate": overall,
            "regression_positive_fold_count": sum(positive),
            "HORIZON_REGRESSION_STABLE": stable,
            "direction_rank_diagnostics": _direction_rank_diagnostics(
                fold_aggregates,  # type: ignore[arg-type]
                overall,
            ),
        }
    winner = select_regression_horizon(horizon_results)
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does forecast horizon affect frozen F0 VN-LSTM out-of-time predictability?",
        "horizons_hours": list(FROZEN_HORIZONS_HOURS),
        "frozen_seeds": list(FROZEN_SEEDS),
        "source_audit": {
            "path": AUDIT_OUTPUT_RELATIVE_PATH.as_posix(),
            "common_anchor_count": EXPECTED_COMMON_COUNT,
            "common_anchor_identity_sha256": EXPECTED_COMMON_IDENTITY,
        },
        "temporal_design": design,
        "target_scaling": {
            "q_formula": "y_t(H) / rolling_volatility_24h_t",
            "c_formula": "RMS_training_scope(q_H) / RMS_training_scope(q_1h)",
            "training_target": "z_t(H) = q_t(H) / c_H",
            "c_1h_exactly": 1.0,
            "sqrt_h_scaling": False,
            "centering": False,
            "clipping": False,
            "epsilon_added": False,
        },
        "horizon_results": {f"{horizon}h": horizon_results[horizon] for horizon in FROZEN_HORIZONS_HOURS},
        "REGRESSION_HORIZON_WINNER": "NONE" if winner is None else f"{winner}h",
        "overlapping_target_warning": (
            "Multi-hour targets overlap; no iid p-values or significance claims are produced."
        ),
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
            "experiment_type": "paired_nested_forecast_horizon_walkforward",
            "horizons_hours": list(FROZEN_HORIZONS_HOURS),
            "frozen_seeds": list(FROZEN_SEEDS),
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "feature_set": "F0",
            "feature_names": list(F0_FEATURE_NAMES),
            "temporal_design": design,
            "original_validation": "NOT READ OR USED",
            "test_set": "NOT READ OR USED",
            "data": {
                "canonical_1h": {
                    "path": CANONICAL_1H_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / CANONICAL_1H_RELATIVE_PATH),
                },
                "horizon_audit": {
                    "path": AUDIT_OUTPUT_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / AUDIT_OUTPUT_RELATIVE_PATH),
                    "audit_id": audit["audit_id"],
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
            f"h{member.horizon_hours}_fold_{member.fold}_seed_{member.seed}": {
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
                "type": "paired_nested_forecast_horizon_walkforward",
                "horizons_hours": list(FROZEN_HORIZONS_HOURS),
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
        positions = folds[member.fold - 1].outer_evaluation_positions
        outer_scale = float(member.metrics["training"]["outer_c_h"])  # type: ignore[index]
        prediction_rows.append(
            pd.DataFrame(
                {
                    "horizon_hours": member.horizon_hours,
                    "fold": member.fold,
                    "seed": member.seed,
                    "decision_time": samples.decision_times[positions],
                    "raw_target": samples.raw_targets[member.horizon_hours][positions],
                    VOLATILITY_FEATURE_NAME: samples.sigma[positions],
                    "c_h": outer_scale,
                    "training_target": (
                        samples.normalized_targets[member.horizon_hours][positions]
                        / outer_scale
                    ),
                    "training_target_prediction": member.normalized_predictions,
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
        stem = f"h{member.horizon_hours}_fold_{member.fold}_seed_{member.seed}"
        checkpoint = model_directory / f"{stem}_lstm.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": member.model.state_dict(),
                "experiment_id": EXPERIMENT_ID,
                "horizon_hours": member.horizon_hours,
                "fold": member.fold,
                "seed": member.seed,
                "inner_best_epoch": member.metrics["training"]["inner_best_epoch"],  # type: ignore[index]
                "refit_epoch_count": member.metrics["training"]["refit_epoch_count"],  # type: ignore[index]
                "outer_c_h": member.metrics["training"]["outer_c_h"],  # type: ignore[index]
                "model_config": resolved.model,
                "training_config": resolved.training,
                "feature_names": list(F0_FEATURE_NAMES),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        joblib.dump(member.scaler, scaler_directory / f"{stem}_scaler.joblib")
    return LSTMRunResult(metrics_path=metrics_path, run_directory=run_directory, result=result)
