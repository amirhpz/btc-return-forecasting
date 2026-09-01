from __future__ import annotations

import json
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
from btc_forecasting.baselines.ridge import _retained_target_splits
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES
from btc_forecasting.features.eng52_build import ENG52_OUTPUT_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
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
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_generalization import _correlation
from btc_forecasting.training.lstm_vn_mse import (
    ResolvedVNMSEConfiguration,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)

EXPERIMENT_ID = "E03-VN-S9"
STABLE_CONFIG_RELATIVE_PATH = Path("configs/features/eng52_stable_candidates.yaml")
EXPECTED_STABLE_CANDIDATES = (
    "band_bb_percB_20_2",
    "body_signed_to_tr",
    "bullish_engulf_score",
    "channel_pos_20",
    "close_location_value",
    "ema_gap_atr_20",
    "mom_stoch_rsi_14_14_3",
    "roc_10",
    "up_close_ratio_5",
)
VOLATILITY_FEATURE_NAME = "rolling_volatility_24h"
VOLATILITY_FEATURE_INDEX = F0_FEATURE_NAMES.index(VOLATILITY_FEATURE_NAME)
TARGET_COLUMNS = ("bar_open_time", "decision_time", "target_time", TARGET_COLUMN)
ONE_HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class ResolvedS9Configuration:
    baseline: ResolvedVNMSEConfiguration
    stable_feature_config: dict[str, Any]


@dataclass(frozen=True)
class PairedSplit:
    control: SequenceSamples
    candidate: SequenceSamples
    raw_targets: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class PreparedS9Data:
    train: PairedSplit
    validation: PairedSplit
    redundancy_report: dict[str, object]
    final_eng_additions: tuple[str, ...]


@dataclass(frozen=True)
class TrainedPairMember:
    model: LSTMRegressor
    scaler: RobustScaler
    scaled: ScaledSequenceSamples
    normalized_predictions: np.ndarray
    history: list[dict[str, float | int]]
    training_metadata: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def resolve_s9_configuration(*, project_root: Path) -> ResolvedS9Configuration:
    baseline = resolve_vn_mse_configuration(project_root=project_root)
    feature_set = load_yaml(project_root / STABLE_CONFIG_RELATIVE_PATH)["feature_set"]
    candidates = tuple(feature_set.get("stable_candidate_features", ()))
    if candidates != EXPECTED_STABLE_CANDIDATES:
        raise ValueError(
            "E03-VN-S9 requires the frozen nine TRAIN-screened stable candidates"
        )
    rule = feature_set.get("screening_rule")
    expected_rule = {
        "required_same_sign_blocks": 5,
        "total_chronological_blocks": 6,
        "full_train_sign_must_match_dominant_block_sign": True,
        "minimum_median_absolute_block_spearman": 0.02,
        "minimum_absolute_full_train_spearman": 0.02,
        "otherwise": "WEAK_OR_UNSTABLE",
    }
    if rule != expected_rule:
        raise ValueError("E03-VN-S9 must not change the frozen screening rule")
    if feature_set.get("train_only_scope") != (
        "Frozen TRAIN targets only; validation and TEST target rows are neither read nor used."
    ):
        raise ValueError("Stable-candidate config must retain its TRAIN-only scope")
    return ResolvedS9Configuration(
        baseline=baseline,
        stable_feature_config=feature_set,
    )


def _aligned_feature_values(
    rows: pd.DataFrame,
    names: tuple[str, ...],
    anchor_times: pd.DatetimeIndex,
) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(rows["open_time"], utc=True))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be strictly ordered and unique")
    positions = times.get_indexer(anchor_times)
    result = np.full((len(anchor_times), len(names)), np.nan, dtype=np.float64)
    present = positions >= 0
    result[present] = rows.loc[:, list(names)].to_numpy(np.float64)[positions[present]]
    return result


def _deterministic_equivalence(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[bool, bool, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(np.count_nonzero(mask))
    if count < 2:
        return False, False, count
    x = left[mask]
    y = right[mask]
    exact = bool(np.allclose(x, y, rtol=1e-12, atol=1e-12))
    if np.all(x == x[0]):
        affine = bool(np.allclose(y, y[0], rtol=1e-12, atol=1e-12))
    else:
        design = np.column_stack((x, np.ones_like(x)))
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (slope * x + intercept)
        scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
        affine = bool(np.max(np.abs(residual)) <= 1e-10 * scale)
    return exact, affine, count


def detect_f0_cross_redundancy(
    f0_rows: pd.DataFrame,
    eng52_rows: pd.DataFrame,
    train_anchor_times: pd.DatetimeIndex,
    *,
    stable_candidates: tuple[str, ...] = EXPECTED_STABLE_CANDIDATES,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Resolve cross-feature redundancy from TRAIN feature values; no target is accepted."""
    if not set(stable_candidates).issubset(ENG52_FEATURE_NAMES):
        raise ValueError("Stable candidate list contains an unknown ENG52 feature")
    f0_values = _aligned_feature_values(f0_rows, F0_FEATURE_NAMES, train_anchor_times)
    eng_values = _aligned_feature_values(eng52_rows, stable_candidates, train_anchor_times)
    exclusions: list[dict[str, object]] = []
    for eng_index, eng_name in enumerate(stable_candidates):
        matches: list[dict[str, object]] = []
        for f0_index, f0_name in enumerate(F0_FEATURE_NAMES):
            eng = eng_values[:, eng_index]
            f0 = f0_values[:, f0_index]
            finite = np.isfinite(eng) & np.isfinite(f0)
            exact, affine, count = _deterministic_equivalence(eng, f0)
            spearman = (
                _correlation(eng[finite], f0[finite], rank=True)
                if count >= 2
                else None
            )
            redundant = exact or affine or (
                spearman is not None and abs(spearman) >= 0.98
            )
            if redundant:
                matches.append(
                    {
                        "matching_f0_feature": f0_name,
                        "pairwise_finite_train_count": count,
                        "exact_equivalence": exact,
                        "deterministic_affine_equivalence": affine,
                        "spearman": spearman,
                    }
                )
        if matches:
            matches.sort(
                key=lambda item: (
                    not bool(item["exact_equivalence"]),
                    not bool(item["deterministic_affine_equivalence"]),
                    -abs(float(item["spearman"])) if item["spearman"] is not None else np.inf,
                    str(item["matching_f0_feature"]),
                )
            )
            chosen = matches[0]
            exclusions.append(
                {
                    "excluded_eng_feature": eng_name,
                    **chosen,
                    "all_redundant_f0_matches": matches,
                }
            )
    excluded = {str(item["excluded_eng_feature"]) for item in exclusions}
    additions = tuple(name for name in stable_candidates if name not in excluded)
    report: dict[str, object] = {
        "scope": "frozen TRAIN feature values only; target values are not inputs",
        "original_stable_candidate_count": len(stable_candidates),
        "original_stable_candidates": list(stable_candidates),
        "redundancy_rule": {
            "exact_numeric_equivalence": True,
            "deterministic_affine_equivalence": True,
            "minimum_absolute_spearman": 0.98,
            "incumbent_on_redundancy": "F0",
        },
        "excluded_eng_features": exclusions,
        "final_eng_additions": list(additions),
        "final_total_input_feature_count": len(F0_FEATURE_NAMES) + len(additions),
    }
    return report, additions


def build_common_paired_split(
    feature_rows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    eng_additions: tuple[str, ...],
) -> PairedSplit:
    feature_names = (*F0_FEATURE_NAMES, *eng_additions)
    required = {"open_time", *feature_names}
    missing = sorted(required - set(feature_rows.columns))
    if missing:
        raise ValueError(f"Missing paired feature columns: {missing}")
    missing_targets = sorted(set(TARGET_COLUMNS) - set(targets.columns))
    if missing_targets:
        raise ValueError(f"Missing paired target columns: {missing_targets}")

    feature_time = pd.DatetimeIndex(pd.to_datetime(feature_rows["open_time"], utc=True))
    if feature_time.has_duplicates or not feature_time.is_monotonic_increasing:
        raise ValueError("Paired feature timestamps must be strictly ordered and unique")
    ordered = targets.sort_values("decision_time", kind="stable").reset_index(drop=True)
    bar_time = pd.DatetimeIndex(pd.to_datetime(ordered["bar_open_time"], utc=True))
    decision_time = pd.DatetimeIndex(pd.to_datetime(ordered["decision_time"], utc=True))
    if not (decision_time - bar_time == ONE_HOUR).all():
        raise ValueError("Paired feature anchors must be available exactly one hour later")

    positions = feature_time.get_indexer(bar_time)
    values = feature_rows.loc[:, list(feature_names)].to_numpy(np.float64)
    raw = ordered[TARGET_COLUMN].to_numpy(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Paired raw targets must be finite")
    windows: list[np.ndarray] = []
    raw_targets: list[float] = []
    normalized_targets: list[float] = []
    sigma_values: list[float] = []
    decisions: list[pd.Timestamp] = []
    expected_span = (LOOKBACK_HOURS - 1) * ONE_HOUR
    for target_index, end in enumerate(positions):
        start = end - LOOKBACK_HOURS + 1
        if end < 0 or start < 0 or feature_time[end] - feature_time[start] != expected_span:
            continue
        window = values[start : end + 1]
        if window.shape != (LOOKBACK_HOURS, len(feature_names)) or not np.isfinite(window).all():
            continue
        sigma = float(window[-1, VOLATILITY_FEATURE_INDEX])
        if not np.isfinite(sigma) or sigma == 0.0:
            continue
        if sigma < 0.0:
            raise ValueError("rolling_volatility_24h cannot be negative")
        windows.append(window)
        raw_targets.append(float(raw[target_index]))
        normalized_targets.append(float(raw[target_index] / sigma))
        sigma_values.append(sigma)
        decisions.append(decision_time[target_index])
    if not windows:
        raise ValueError("No common paired samples satisfy the frozen eligibility rules")

    candidate_features = np.stack(windows).astype(np.float32, copy=False)
    normalized = np.asarray(normalized_targets, dtype=np.float64)
    common_decisions = pd.DatetimeIndex(decisions)
    excluded = len(ordered) - len(windows)

    def samples(features: np.ndarray) -> SequenceSamples:
        return SequenceSamples(
            features=features,
            targets=normalized,
            decision_times=common_decisions,
            candidate_count=len(ordered),
            excluded_lookback_count=excluded,
        )

    control = samples(candidate_features[:, :, : len(F0_FEATURE_NAMES)].copy())
    candidate = samples(candidate_features)
    if _sample_identity(control.decision_times) != _sample_identity(candidate.decision_times):
        raise RuntimeError("Paired models must use identical sample identities")
    return PairedSplit(
        control=control,
        candidate=candidate,
        raw_targets=np.asarray(raw_targets, dtype=np.float64),
        sigma=np.asarray(sigma_values, dtype=np.float64),
    )


def _load_development_targets(
    target_path: Path,
    *,
    split_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = _parse_utc(
        split_metadata["split"]["boundaries"]["test"][
            "decision_time_start_inclusive"
        ]
    )
    rows = pq.read_table(
        target_path,
        columns=list(TARGET_COLUMNS),
        filters=[("decision_time", "<", test_start)],
    ).to_pandas()
    train, validation = _retained_target_splits(rows, split_metadata=split_metadata)
    expected = split_metadata["split"]["retained_rows"]
    if len(train) != int(expected["train"]) or len(validation) != int(expected["validation"]):
        raise ValueError("Development target counts do not match the frozen split")
    return train, validation


def prepare_s9_data(
    *,
    project_root: Path,
    resolved: ResolvedS9Configuration | None = None,
) -> PreparedS9Data:
    root = project_root.resolve()
    configuration = resolved or resolve_s9_configuration(project_root=root)
    split_metadata = load_yaml(root / FROZEN_SPLIT_RELATIVE_PATH)
    train_targets, validation_targets = _load_development_targets(
        root / TARGET_RELATIVE_PATH,
        split_metadata=split_metadata,
    )
    maximum_anchor = pd.to_datetime(validation_targets["bar_open_time"], utc=True).max()
    canonical = pq.read_table(
        root / CANONICAL_1H_RELATIVE_PATH,
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    ).to_pandas()
    eng52 = pq.read_table(
        root / ENG52_OUTPUT_RELATIVE_PATH,
        columns=["open_time", *EXPECTED_STABLE_CANDIDATES],
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    ).to_pandas()
    f0 = compute_f0_features(canonical)
    train_anchors = pd.DatetimeIndex(
        pd.to_datetime(train_targets["bar_open_time"], utc=True)
    )
    redundancy, additions = detect_f0_cross_redundancy(
        f0,
        eng52,
        train_anchors,
        stable_candidates=tuple(
            configuration.stable_feature_config["stable_candidate_features"]
        ),
    )
    combined = f0.merge(eng52, on="open_time", how="inner", validate="one_to_one")
    train = build_common_paired_split(combined, train_targets, eng_additions=additions)
    validation = build_common_paired_split(
        combined,
        validation_targets,
        eng_additions=additions,
    )
    return PreparedS9Data(
        train=train,
        validation=validation,
        redundancy_report=redundancy,
        final_eng_additions=additions,
    )


def fit_generic_train_scaler(
    train: SequenceSamples,
    validation: SequenceSamples,
) -> ScaledSequenceSamples:
    if train.features.ndim != 3 or validation.features.ndim != 3:
        raise ValueError("Paired LSTM features must be three-dimensional sequences")
    if train.features.shape[1:] != validation.features.shape[1:]:
        raise ValueError("Paired train and validation feature shapes must agree")
    feature_count = train.features.shape[2]
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
    )
    train_scaled = scaler.fit_transform(train.features.reshape(-1, feature_count)).reshape(
        train.features.shape
    )
    validation_scaled = scaler.transform(
        validation.features.reshape(-1, feature_count)
    ).reshape(validation.features.shape)

    def replace(source: SequenceSamples, features: np.ndarray) -> SequenceSamples:
        return SequenceSamples(
            features=features.astype(np.float32, copy=False),
            targets=source.targets,
            decision_times=source.decision_times,
            candidate_count=source.candidate_count,
            excluded_lookback_count=source.excluded_lookback_count,
        )

    return ScaledSequenceSamples(
        train=replace(train, train_scaled),
        validation=replace(validation, validation_scaled),
        feature_scaler=scaler,
    )


def _train_member(
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    resolved: ResolvedS9Configuration,
    seed: int,
    device: torch.device,
) -> TrainedPairMember:
    configure_determinism(seed)
    scaled = fit_generic_train_scaler(train, validation)
    model = LSTMRegressor(
        input_size=train.features.shape[2],
        hidden_size=int(resolved.baseline.model["hidden_size"]),
        num_layers=int(resolved.baseline.model["num_layers"]),
        configured_dropout=float(resolved.baseline.model["dropout"]),
    )
    outcome = fit_lstm(
        model,
        train=scaled.train,
        validation=scaled.validation,
        training_config=resolved.baseline.training,
        seed=seed,
        device=device,
    )
    predictions = predict_lstm(
        outcome.model,
        scaled.validation.features,
        batch_size=int(resolved.baseline.training["batch_size"]),
        device=device,
    )
    return TrainedPairMember(
        model=outcome.model,
        scaler=scaled.feature_scaler,
        scaled=scaled,
        normalized_predictions=predictions,
        history=outcome.history,
        training_metadata={
            "epochs_trained": outcome.epochs_trained,
            "best_epoch": outcome.best_epoch,
            "best_validation_normalized_target_mse_loss": outcome.best_validation_loss,
            "duration_seconds": outcome.duration_seconds,
            "configured_dropout": model.configured_dropout,
            "effective_lstm_dropout": model.effective_lstm_dropout,
        },
    )


def evaluate_pair_member(
    split: PairedSplit,
    normalized_predictions: np.ndarray,
) -> dict[str, object]:
    raw_predictions = reconstruct_raw_predictions(normalized_predictions, split.sigma)
    raw_metrics = regression_metrics(split.raw_targets, raw_predictions)
    zero_metrics = regression_loss_metrics(
        split.raw_targets,
        zero_return_prediction(len(split.raw_targets)),
    )
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    normalized_targets = split.control.targets
    predictions = np.asarray(normalized_predictions, dtype=np.float64)
    return {
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
            "spearman_rank_ic": _correlation(normalized_targets, predictions, rank=True),
            "prediction_std": float(np.std(predictions, ddof=0)),
            "target_std": float(np.std(normalized_targets, ddof=0)),
        },
        "raw_predictions": raw_predictions,
    }


def paired_metric_deltas(
    control: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    groups: dict[str, tuple[str, ...]] = {
        "raw_return_validation": (
            "mae",
            "rmse",
            "r2",
            "pearson_ic",
            "spearman_rank_ic",
            "directional_accuracy",
        ),
        "same_row_zero_return": ("mae", "rmse"),
        "skill": ("mae", "rmse"),
        "normalized_space": (
            "mse",
            "pearson_ic",
            "spearman_rank_ic",
            "prediction_std",
            "target_std",
        ),
    }
    result: dict[str, object] = {}
    for group, names in groups.items():
        control_group = control[group]
        candidate_group = candidate[group]
        assert isinstance(control_group, dict) and isinstance(candidate_group, dict)
        result[group] = {
            name: float(candidate_group[name]) - float(control_group[name])
            for name in names
        }
    return result


def _without_predictions(evaluation: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in evaluation.items() if key != "raw_predictions"}


def run_lstm_vn_stable_paired_experiment(*, project_root: Path) -> LSTMRunResult:
    root = project_root.resolve()
    resolved = resolve_s9_configuration(project_root=root)
    prepared = prepare_s9_data(project_root=root, resolved=resolved)
    device_info = require_official_cuda()
    seed = int(resolved.baseline.experiment["seed"])

    control = _train_member(
        train=prepared.train.control,
        validation=prepared.validation.control,
        resolved=resolved,
        seed=seed,
        device=device_info.device,
    )
    candidate = _train_member(
        train=prepared.train.candidate,
        validation=prepared.validation.candidate,
        resolved=resolved,
        seed=seed,
        device=device_info.device,
    )
    control_evaluation = evaluate_pair_member(
        prepared.validation,
        control.normalized_predictions,
    )
    candidate_evaluation = evaluate_pair_member(
        prepared.validation,
        candidate.normalized_predictions,
    )
    train_identity = _sample_identity(prepared.train.control.decision_times)
    validation_identity = _sample_identity(prepared.validation.control.decision_times)
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "question": (
            "Do TRAIN-selected stable ENG52 features add predictive value beyond F0?"
        ),
        "controlled_difference": "input feature set only",
        "f0_cross_redundancy": prepared.redundancy_report,
        "resolved_feature_sets": {
            "paired_control": list(F0_FEATURE_NAMES),
            "candidate": [*F0_FEATURE_NAMES, *prepared.final_eng_additions],
            "final_eng_additions": list(prepared.final_eng_additions),
        },
        "common_samples": {
            "train": {
                "candidate_rows": prepared.train.control.candidate_count,
                "eligible_samples": len(prepared.train.control.targets),
                "sample_identity_sha256": train_identity,
            },
            "validation": {
                "candidate_rows": prepared.validation.control.candidate_count,
                "eligible_samples": len(prepared.validation.control.targets),
                "sample_identity_sha256": validation_identity,
            },
            "paired_control_and_candidate_identical": True,
        },
        "paired_control": _without_predictions(control_evaluation),
        "candidate": _without_predictions(candidate_evaluation),
        "candidate_minus_paired_control": paired_metric_deltas(
            control_evaluation,
            candidate_evaluation,
        ),
        "training": {
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "seed": seed,
            "paired_control": control.training_metadata,
            "candidate": candidate.training_metadata,
        },
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
    }

    run_id = f"{EXPERIMENT_ID}_1h_F0S9_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.parquet"
    history_path = run_directory / "training_history.json"
    resolved_config_path = run_directory / "resolved_config.json"

    artifact_paths = {
        "paired_control_checkpoint": model_directory / "paired_control_lstm.pt",
        "candidate_checkpoint": model_directory / "candidate_lstm.pt",
        "paired_control_scaler": scaler_directory / "paired_control_scaler.joblib",
        "candidate_scaler": scaler_directory / "candidate_scaler.joblib",
    }
    manifest = create_run_manifest(
        project_root=root,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
    )
    manifest.update(
        {
            "controlled_difference": "input feature set only",
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "seed": seed,
            "model_config": resolved.baseline.model,
            "training_config": resolved.baseline.training,
            "resolved_feature_sets": result["resolved_feature_sets"],
            "common_samples": result["common_samples"],
            "evaluated_splits": ["validation"],
            "test_set": "NOT EVALUATED",
            "data": {
                "canonical_1h": {
                    "path": CANONICAL_1H_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / CANONICAL_1H_RELATIVE_PATH),
                },
                "eng52": {
                    "path": ENG52_OUTPUT_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / ENG52_OUTPUT_RELATIVE_PATH),
                },
                "target": {
                    "path": TARGET_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / TARGET_RELATIVE_PATH),
                },
                "stable_candidates": {
                    "path": STABLE_CONFIG_RELATIVE_PATH.as_posix(),
                    "sha256": sha256_file(root / STABLE_CONFIG_RELATIVE_PATH),
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
                **{
                    name: path.relative_to(root).as_posix()
                    for name, path in artifact_paths.items()
                },
            },
        }
    )
    write_manifest(run_directory / "manifest.json", manifest)
    _write_json(metrics_path, result)
    _write_json(
        history_path,
        {
            "paired_control": control.history,
            "candidate": candidate.history,
        },
    )
    _write_json(
        resolved_config_path,
        {
            "experiment": {
                "id": EXPERIMENT_ID,
                "base_experiment": "E03-VN-MSE",
                "controlled_difference": "input feature set only",
            },
            "stable_feature_config": resolved.stable_feature_config,
            "f0_cross_redundancy": prepared.redundancy_report,
            "resolved_feature_sets": result["resolved_feature_sets"],
            "model": resolved.baseline.model,
            "training": resolved.baseline.training,
        },
    )
    predictions = pd.DataFrame(
        {
            "decision_time": prepared.validation.control.decision_times,
            TARGET_COLUMN: prepared.validation.raw_targets,
            VOLATILITY_FEATURE_NAME: prepared.validation.sigma,
            "normalized_target": prepared.validation.control.targets,
            "paired_control_normalized_prediction": control.normalized_predictions,
            "candidate_normalized_prediction": candidate.normalized_predictions,
            "paired_control_raw_return_prediction": control_evaluation["raw_predictions"],
            "candidate_raw_return_prediction": candidate_evaluation["raw_predictions"],
            "zero_return_prediction": zero_return_prediction(
                len(prepared.validation.raw_targets)
            ),
        }
    )
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    predictions.to_parquet(temporary_predictions, index=False)
    os.replace(temporary_predictions, predictions_path)

    model_directory.mkdir(parents=True, exist_ok=False)
    scaler_directory.mkdir(parents=True, exist_ok=False)
    for name, member in (("paired_control", control), ("candidate", candidate)):
        checkpoint = artifact_paths[f"{name}_checkpoint"]
        temporary = checkpoint.with_suffix(".pt.tmp")
        feature_names = (
            list(F0_FEATURE_NAMES)
            if name == "paired_control"
            else [*F0_FEATURE_NAMES, *prepared.final_eng_additions]
        )
        torch.save(
            {
                "model_state_dict": member.model.state_dict(),
                "experiment_id": EXPERIMENT_ID,
                "pair_member": name,
                "model_config": resolved.baseline.model,
                "training_config": resolved.baseline.training,
                "feature_names": feature_names,
                "target_definition": "z_t = raw_return / rolling_volatility_24h_t",
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        joblib.dump(member.scaler, artifact_paths[f"{name}_scaler"])
    return LSTMRunResult(
        metrics_path=metrics_path,
        run_directory=run_directory,
        result=result,
    )
