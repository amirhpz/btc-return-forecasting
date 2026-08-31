from __future__ import annotations

import hashlib
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH

F0_CONFIG_RELATIVE_PATH = Path("configs/features/f0_minimal.yaml")
RIDGE_CONFIG_RELATIVE_PATH = Path("configs/models/ridge.yaml")
E02_CONFIG_RELATIVE_PATH = Path("configs/experiments/e02.yaml")
LOOKBACK_HOURS = 24
ONE_HOUR = timedelta(hours=1)
CANONICAL_FEATURE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")
TARGET_COLUMNS = ("bar_open_time", "decision_time", "target_time", TARGET_COLUMN)


@dataclass(frozen=True)
class RidgeSamples:
    features: np.ndarray
    targets: np.ndarray
    decision_times: pd.DatetimeIndex
    candidate_count: int
    excluded_lookback_count: int


@dataclass(frozen=True)
class PreparedDevelopmentSamples:
    train: RidgeSamples
    validation: RidgeSamples


@dataclass(frozen=True)
class TrainedRidge:
    model: Ridge
    feature_scaler: RobustScaler
    target_scaler: StandardScaler

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled_features = self.feature_scaler.transform(features)
        scaled_predictions = self.model.predict(scaled_features).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled_predictions).reshape(-1)


@dataclass(frozen=True)
class RidgeEvaluation:
    result: dict[str, object]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray


@dataclass(frozen=True)
class RidgeRunResult:
    metrics_path: Path
    predictions_path: Path
    result: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _sample_identity(decision_times: pd.DatetimeIndex) -> str:
    values = decision_times.asi8.astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _requested_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    metrics = regression_metrics(targets, predictions)
    return {
        key: metrics[key]
        for key in (
            "n",
            "mae",
            "rmse",
            "r2",
            "pearson_ic",
            "spearman_rank_ic",
            "directional_accuracy",
        )
    }


def build_lookback_samples(
    feature_rows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    lookback_hours: int = LOOKBACK_HOURS,
) -> RidgeSamples:
    """Build oldest-to-newest flattened windows without crossing an hourly gap."""
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")
    required_feature_columns = {"open_time", *F0_FEATURE_NAMES}
    missing_features = sorted(required_feature_columns - set(feature_rows.columns))
    if missing_features:
        raise ValueError(f"Missing F0 feature columns: {missing_features}")
    missing_targets = sorted(set(TARGET_COLUMNS) - set(targets.columns))
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    feature_time = pd.DatetimeIndex(pd.to_datetime(feature_rows["open_time"], utc=True))
    if feature_time.has_duplicates or not feature_time.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be strictly ordered and unique")
    ordered_targets = targets.sort_values("decision_time", kind="stable").reset_index(drop=True)
    bar_time = pd.DatetimeIndex(pd.to_datetime(ordered_targets["bar_open_time"], utc=True))
    decision_time = pd.DatetimeIndex(
        pd.to_datetime(ordered_targets["decision_time"], utc=True)
    )
    if not (decision_time - bar_time == ONE_HOUR).all():
        raise ValueError("Each feature anchor must become available exactly one hour later")

    feature_values = feature_rows.loc[:, list(F0_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    target_values = ordered_targets[TARGET_COLUMN].to_numpy(dtype=np.float64)
    positions = feature_time.get_indexer(bar_time)
    windows: list[np.ndarray] = []
    eligible_targets: list[float] = []
    eligible_decisions: list[pd.Timestamp] = []
    excluded = 0
    expected_span = (lookback_hours - 1) * ONE_HOUR

    for target_position, feature_position in enumerate(positions):
        start = feature_position - lookback_hours + 1
        if feature_position < 0 or start < 0:
            excluded += 1
            continue
        if feature_time[feature_position] - feature_time[start] != expected_span:
            excluded += 1
            continue
        window = feature_values[start : feature_position + 1]
        if window.shape != (lookback_hours, len(F0_FEATURE_NAMES)):
            excluded += 1
            continue
        if not np.isfinite(window).all() or not np.isfinite(target_values[target_position]):
            excluded += 1
            continue
        windows.append(window.reshape(-1))
        eligible_targets.append(float(target_values[target_position]))
        eligible_decisions.append(decision_time[target_position])

    feature_count = lookback_hours * len(F0_FEATURE_NAMES)
    matrix = (
        np.vstack(windows).astype(np.float64, copy=False)
        if windows
        else np.empty((0, feature_count), dtype=np.float64)
    )
    return RidgeSamples(
        features=matrix,
        targets=np.asarray(eligible_targets, dtype=np.float64),
        decision_times=pd.DatetimeIndex(eligible_decisions),
        candidate_count=len(ordered_targets),
        excluded_lookback_count=excluded,
    )


def fit_ridge_model(
    train: RidgeSamples,
    *,
    alpha: float,
    fit_intercept: bool,
) -> TrainedRidge:
    """Fit both scalers and Ridge using train samples only."""
    if len(train.targets) == 0:
        raise ValueError("Ridge training samples must not be empty")
    feature_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
    )
    target_scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_features = feature_scaler.fit_transform(train.features)
    scaled_targets = target_scaler.fit_transform(train.targets.reshape(-1, 1)).reshape(-1)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(scaled_features, scaled_targets)
    return TrainedRidge(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def evaluate_ridge_baseline(
    trained: TrainedRidge,
    *,
    train: RidgeSamples,
    validation: RidgeSamples,
) -> RidgeEvaluation:
    """Evaluate Ridge and zero on the same development rows; never accept test rows."""
    train_predictions = trained.predict(train.features)
    validation_predictions = trained.predict(validation.features)
    zero_predictions = zero_return_prediction(len(validation.targets))
    train_metrics = _requested_metrics(train.targets, train_predictions)
    validation_metrics = _requested_metrics(validation.targets, validation_predictions)
    zero_metrics = regression_loss_metrics(validation.targets, zero_predictions)
    identity = _sample_identity(validation.decision_times)
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    result: dict[str, object] = {
        "experiment_id": "E02",
        "model": "ridge",
        "evaluated_splits": ["train", "validation"],
        "data": {
            "lookback_hours": LOOKBACK_HOURS,
            "window_shape": [LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
            "flattened_feature_count": LOOKBACK_HOURS * len(F0_FEATURE_NAMES),
            "train_candidate_rows": train.candidate_count,
            "train_eligible_samples": len(train.targets),
            "train_excluded_insufficient_or_nonconsecutive_lookback": (
                train.excluded_lookback_count
            ),
            "validation_candidate_rows": validation.candidate_count,
            "validation_eligible_samples": len(validation.targets),
            "validation_excluded_insufficient_or_nonconsecutive_lookback": (
                validation.excluded_lookback_count
            ),
            "total_excluded_insufficient_or_nonconsecutive_lookback": (
                train.excluded_lookback_count + validation.excluded_lookback_count
            ),
        },
        "ridge": {
            "train": train_metrics,
            "validation": {
                **validation_metrics,
                "sample_identity_sha256": identity,
            },
        },
        "zero_return_same_validation_rows": {
            "n": int(zero_metrics["n"]),
            "mae": zero_mae,
            "rmse": zero_rmse,
            "sample_identity_sha256": identity,
        },
        "skill": {
            "mae": 1.0 - float(validation_metrics["mae"]) / zero_mae,
            "rmse": 1.0 - float(validation_metrics["rmse"]) / zero_rmse,
        },
        "test_set": "NOT EVALUATED",
    }
    return RidgeEvaluation(
        result=result,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
    )


def _retained_target_splits(
    target_rows: pd.DataFrame,
    *,
    split_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = split_metadata["split"]["boundaries"]
    train_start = _parse_utc(boundaries["train"]["decision_time_start_inclusive"])
    validation_start = _parse_utc(
        boundaries["validation"]["decision_time_start_inclusive"]
    )
    test_start = _parse_utc(boundaries["test"]["decision_time_start_inclusive"])
    train_target_end = _parse_utc(boundaries["train"]["target_time_end_exclusive"])
    validation_target_end = _parse_utc(
        boundaries["validation"]["target_time_end_exclusive"]
    )
    decision = pd.to_datetime(target_rows["decision_time"], utc=True)
    target_time = pd.to_datetime(target_rows["target_time"], utc=True)
    train = target_rows.loc[
        (decision >= train_start)
        & (decision < validation_start)
        & (target_time < train_target_end)
    ].copy()
    validation = target_rows.loc[
        (decision >= validation_start)
        & (decision < test_start)
        & (target_time < validation_target_end)
    ].copy()
    return train, validation


def load_development_samples(
    *,
    canonical_path: Path,
    target_path: Path,
    split_metadata_path: Path,
) -> PreparedDevelopmentSamples:
    """Read and prepare retained train/validation rows without reading test targets."""
    split_metadata = load_yaml(split_metadata_path)
    boundaries = split_metadata["split"]["boundaries"]
    test_start = _parse_utc(boundaries["test"]["decision_time_start_inclusive"])
    target_table = pq.read_table(
        target_path,
        columns=list(TARGET_COLUMNS),
        filters=[("decision_time", "<", test_start)],
    )
    target_rows = target_table.to_pandas()
    train_targets, validation_targets = _retained_target_splits(
        target_rows,
        split_metadata=split_metadata,
    )
    expected = split_metadata["split"]["retained_rows"]
    if len(train_targets) != expected["train"] or len(validation_targets) != expected["validation"]:
        raise ValueError(
            "Development split counts do not match frozen metadata: "
            f"train={len(train_targets)}, validation={len(validation_targets)}"
        )
    maximum_anchor = pd.to_datetime(validation_targets["bar_open_time"], utc=True).max()
    canonical_table = pq.read_table(
        canonical_path,
        columns=list(CANONICAL_FEATURE_COLUMNS),
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    )
    feature_rows = compute_f0_features(canonical_table.to_pandas())
    return PreparedDevelopmentSamples(
        train=build_lookback_samples(feature_rows, train_targets),
        validation=build_lookback_samples(feature_rows, validation_targets),
    )


def _load_resolved_configuration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_config = load_yaml(root / F0_CONFIG_RELATIVE_PATH)["feature_set"]
    model_config = load_yaml(root / RIDGE_CONFIG_RELATIVE_PATH)["model"]
    experiment_config = load_yaml(root / E02_CONFIG_RELATIVE_PATH)["experiment"]
    if tuple(feature_config["features"]) != F0_FEATURE_NAMES:
        raise ValueError("Frozen F0 config does not match the implemented feature order")
    required_model_values = {
        "family": "ridge_regression",
        "input_view": "flattened_window",
        "feature_scaling": "train_only_robust",
        "target_scaling": "train_only_standard",
    }
    mismatches = {
        key: model_config.get(key)
        for key, expected in required_model_values.items()
        if model_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported frozen Ridge configuration: {mismatches}")
    if experiment_config.get("lookback") != "24h":
        raise ValueError("E02 frozen lookback must be 24h")
    return feature_config, model_config


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def run_ridge_baseline(*, project_root: Path) -> RidgeRunResult:
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
    trained = fit_ridge_model(
        samples.train,
        alpha=float(model_config["alpha"]),
        fit_intercept=bool(model_config["fit_intercept"]),
    )
    evaluation = evaluate_ridge_baseline(
        trained,
        train=samples.train,
        validation=samples.validation,
    )

    run_id = f"E02_1h_F0_B1_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    predictions_path = run_directory / "predictions.parquet"
    metrics_path = run_directory / "metrics.json"
    model_path = model_directory / "ridge.joblib"
    feature_scaler_path = scaler_directory / "feature_scaler.joblib"
    target_scaler_path = scaler_directory / "target_scaler.joblib"

    manifest = create_run_manifest(project_root=root, experiment_id="E02", run_id=run_id)
    manifest.update(
        {
            "model": model_config,
            "timeframe": "1h",
            "lookback": "24h",
            "horizon": "1h",
            "feature_set": feature_config["id"],
            "feature_names": list(F0_FEATURE_NAMES),
            "window_flattening_order": "oldest_to_newest_then_f0_config_order",
            "evaluated_splits": ["train", "validation"],
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
    validation_predictions = pd.DataFrame(
        {
            "decision_time": samples.validation.decision_times,
            TARGET_COLUMN: samples.validation.targets,
            "ridge_prediction": evaluation.validation_predictions,
            "zero_return_prediction": zero_return_prediction(len(samples.validation.targets)),
        }
    )
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    validation_predictions.to_parquet(temporary_predictions, index=False)
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
