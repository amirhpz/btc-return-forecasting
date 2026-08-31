from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.baselines.ridge import (
    CANONICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _parse_utc,
    _retained_target_splits,
)
from btc_forecasting.common.config import load_yaml
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.training.lstm import (
    SequenceSamples,
    build_sequence_samples,
    build_training_components,
    configure_determinism,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)

OVERFIT_SAMPLE_COUNT = 512
OVERFIT_MAX_EPOCHS = 300


@dataclass(frozen=True)
class SourceRun:
    run_directory: Path
    manifest: dict[str, Any]
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    checkpoint_path: Path
    scaler_path: Path
    canonical_path: Path
    target_path: Path
    split_path: Path


@dataclass(frozen=True)
class DiagnosticRunResult:
    artifact_path: Path
    result: dict[str, object]


def _resolve_inside(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Diagnostic path escapes the project root: {path}") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_source_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("experiment_id") != "E03":
        raise ValueError("E03-D requires an E03 source run")
    if manifest.get("test_set") != "NOT EVALUATED":
        raise ValueError("E03-D refuses a source run that accessed the test set")
    evaluated = manifest.get("evaluated_splits")
    if evaluated != ["validation"]:
        raise ValueError("E03-D requires the official validation-only E03 run")


def load_source_run(*, project_root: Path, source_run: Path) -> SourceRun:
    root = project_root.resolve()
    run_directory = _resolve_inside(root, source_run)
    manifest = _read_json(run_directory / "manifest.json")
    validate_source_manifest(manifest)
    artifacts = manifest["artifacts"]
    data = manifest["data"]
    model_config = manifest["model_config"]
    training_config = manifest["training_config"]
    if model_config != load_yaml(root / "configs/models/lstm.yaml")["model"]:
        raise ValueError("Current frozen LSTM config differs from the source E03 run")
    if training_config != load_yaml(root / "configs/training.yaml")["training"]:
        raise ValueError("Current frozen training config differs from the source E03 run")

    resolved = SourceRun(
        run_directory=run_directory,
        manifest=manifest,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=_resolve_inside(root, Path(artifacts["checkpoint"])),
        scaler_path=_resolve_inside(root, Path(artifacts["feature_scaler"])),
        canonical_path=_resolve_inside(root, Path(data["canonical_1h"]["path"])),
        target_path=_resolve_inside(root, Path(data["target"]["path"])),
        split_path=_resolve_inside(root, Path(manifest["split"]["path"])),
    )
    required = (
        resolved.checkpoint_path,
        resolved.scaler_path,
        resolved.canonical_path,
        resolved.target_path,
        resolved.split_path,
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing E03-D source artifacts: {missing}")
    return resolved


def _replace_features(samples: SequenceSamples, features: np.ndarray) -> SequenceSamples:
    return SequenceSamples(
        features=features.astype(np.float32, copy=False),
        targets=samples.targets,
        decision_times=samples.decision_times,
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def apply_restored_scaler(
    samples: SequenceSamples,
    scaler: RobustScaler,
) -> SequenceSamples:
    feature_count = len(F0_FEATURE_NAMES)
    flattened = samples.features.reshape(-1, feature_count)
    transformed = scaler.transform(flattened).reshape(samples.features.shape)
    return _replace_features(samples, transformed)


def select_overfit_subset(
    train: SequenceSamples,
    *,
    sample_count: int = OVERFIT_SAMPLE_COUNT,
) -> SequenceSamples:
    if sample_count != OVERFIT_SAMPLE_COUNT:
        raise ValueError(f"E03-D overfit subset must contain exactly {OVERFIT_SAMPLE_COUNT} rows")
    if len(train.targets) < sample_count:
        raise ValueError("Eligible training samples are insufficient for E03-D")
    return SequenceSamples(
        features=train.features[:sample_count].copy(),
        targets=train.targets[:sample_count].copy(),
        decision_times=train.decision_times[:sample_count],
        candidate_count=sample_count,
        excluded_lookback_count=0,
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "positive_ratio": float(np.mean(array > 0.0)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)


def _finite_float(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def diagnostic_statistics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Diagnostic targets and predictions must have the same non-empty shape")
    metrics = regression_metrics(truth, predicted)
    target_distribution = _distribution(truth)
    prediction_distribution = _distribution(predicted)
    residual = truth - predicted
    return {
        "model_metrics": {
            "n": int(metrics["n"]),
            "mae": _finite_float(metrics["mae"]),
            "rmse": _finite_float(metrics["rmse"]),
            "r2": _finite_float(metrics["r2"]),
            "pearson_ic": _finite_float(metrics["pearson_ic"]),
            "spearman_rank_ic": _finite_float(metrics["spearman_rank_ic"]),
            "directional_accuracy": _finite_float(metrics["directional_accuracy"]),
        },
        "target_distribution": target_distribution,
        "prediction_distribution": prediction_distribution,
        "residual": {
            "definition": "target_minus_prediction",
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual, ddof=0)),
        },
        "prediction_std_over_target_std": _safe_ratio(
            prediction_distribution["std"], target_distribution["std"]
        ),
        "mean_abs_prediction_over_mean_abs_target": _safe_ratio(
            float(np.mean(np.abs(predicted))),
            float(np.mean(np.abs(truth))),
        ),
    }


def build_distribution_report(
    *,
    train: SequenceSamples,
    train_predictions: np.ndarray,
    validation: SequenceSamples,
    validation_predictions: np.ndarray,
    source_run_id: str,
    device: str,
    gpu_name: str,
) -> dict[str, object]:
    validation_report = diagnostic_statistics(validation.targets, validation_predictions)
    zero_metrics = regression_loss_metrics(
        validation.targets,
        zero_return_prediction(len(validation.targets)),
    )
    model_metrics = validation_report["model_metrics"]
    assert isinstance(model_metrics, dict)
    if model_metrics["mae"] is None or model_metrics["rmse"] is None:
        raise ValueError("Validation error metrics must be finite")
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    validation_report["zero_return_same_rows"] = {
        "n": int(zero_metrics["n"]),
        "mae": zero_mae,
        "rmse": zero_rmse,
    }
    validation_report["skill"] = {
        "mae": 1.0 - float(model_metrics["mae"]) / zero_mae,
        "rmse": 1.0 - float(model_metrics["rmse"]) / zero_rmse,
    }
    return {
        "diagnostic_id": "E03-D",
        "diagnostic_type": "prediction_distribution",
        "source_run_id": source_run_id,
        "device": device,
        "gpu_name": gpu_name,
        "evaluated_splits": ["train", "validation"],
        "train": diagnostic_statistics(train.targets, train_predictions),
        "validation": validation_report,
        "test_set": "NOT EVALUATED",
    }


def _restore_model(source: SourceRun, *, device: torch.device) -> LSTMRegressor:
    model = LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(source.model_config["hidden_size"]),
        num_layers=int(source.model_config["num_layers"]),
        configured_dropout=float(source.model_config["dropout"]),
    )
    checkpoint = torch.load(source.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _write_diagnostic(
    *,
    root: Path,
    directory_name: str,
    result: dict[str, object],
) -> DiagnosticRunResult:
    output_directory = root / "outputs" / "diagnostics" / directory_name
    output_directory.mkdir(parents=True, exist_ok=False)
    artifact_path = output_directory / "diagnostic.json"
    temporary_path = artifact_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, artifact_path)
    return DiagnosticRunResult(artifact_path=artifact_path, result=result)


def run_prediction_distribution_diagnostic(
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
        raise TypeError("E03-D source scaler is not a RobustScaler")
    train = apply_restored_scaler(prepared.train, scaler)
    validation = apply_restored_scaler(prepared.validation, scaler)
    model = _restore_model(source, device=device_info.device)
    batch_size = int(source.training_config["batch_size"])
    train_predictions = predict_lstm(
        model,
        train.features,
        batch_size=batch_size,
        device=device_info.device,
    )
    validation_predictions = predict_lstm(
        model,
        validation.features,
        batch_size=batch_size,
        device=device_info.device,
    )
    result = build_distribution_report(
        train=train,
        train_predictions=train_predictions,
        validation=validation,
        validation_predictions=validation_predictions,
        source_run_id=str(source.manifest["run_id"]),
        device=str(device_info.device),
        gpu_name=device_info.gpu_name,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03D_lstm_distribution_{timestamp}",
        result=result,
    )


def _load_train_only_samples(source: SourceRun) -> SequenceSamples:
    split_metadata = load_yaml(source.split_path)
    boundaries = split_metadata["split"]["boundaries"]
    validation_start = _parse_utc(
        boundaries["validation"]["decision_time_start_inclusive"]
    )
    target_table = pq.read_table(
        source.target_path,
        columns=list(TARGET_COLUMNS),
        filters=[("decision_time", "<", validation_start)],
    )
    train_targets, _ = _retained_target_splits(
        target_table.to_pandas(),
        split_metadata=split_metadata,
    )
    expected_train = int(split_metadata["split"]["retained_rows"]["train"])
    if len(train_targets) != expected_train:
        raise ValueError(
            f"Training rows do not match frozen metadata: {len(train_targets)} != {expected_train}"
        )
    maximum_anchor = pd.to_datetime(train_targets["bar_open_time"], utc=True).max()
    canonical_table = pq.read_table(
        source.canonical_path,
        columns=list(CANONICAL_FEATURE_COLUMNS),
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    )
    feature_rows = compute_f0_features(canonical_table.to_pandas())
    return build_sequence_samples(feature_rows, train_targets)


def _loss_and_error_metrics(
    model: LSTMRegressor,
    samples: SequenceSamples,
    *,
    loss_function: torch.nn.HuberLoss,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    predictions = predict_lstm(
        model,
        samples.features,
        batch_size=batch_size,
        device=device,
    )
    loss = loss_function(
        torch.from_numpy(predictions.astype(np.float32, copy=False)),
        torch.from_numpy(samples.targets.astype(np.float32, copy=False)),
    )
    metrics = regression_loss_metrics(samples.targets, predictions)
    return float(loss.item()), float(metrics["mae"]), float(metrics["rmse"])


def run_overfit_sanity_diagnostic(
    *,
    project_root: Path,
    source_run: Path,
) -> DiagnosticRunResult:
    root = project_root.resolve()
    source = load_source_run(project_root=root, source_run=source_run)
    device_info = require_official_cuda()
    seed = int(source.manifest["seed"])
    configure_determinism(seed)
    train = _load_train_only_samples(source)
    scaler = joblib.load(source.scaler_path)
    if not isinstance(scaler, RobustScaler):
        raise TypeError("E03-D source scaler is not a RobustScaler")
    subset = select_overfit_subset(apply_restored_scaler(train, scaler))
    model = LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(source.model_config["hidden_size"]),
        num_layers=int(source.model_config["num_layers"]),
        configured_dropout=float(source.model_config["dropout"]),
    ).to(device_info.device)
    optimizer, loss_function, _ = build_training_components(
        model,
        training_config=source.training_config,
    )
    batch_size = int(source.training_config["batch_size"])
    initial_loss, initial_mae, initial_rmse = _loss_and_error_metrics(
        model,
        subset,
        loss_function=loss_function,
        batch_size=batch_size,
        device=device_info.device,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(subset.features),
            torch.from_numpy(subset.targets.astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    max_norm = float(source.training_config["gradient_clipping"]["max_norm"])
    norm_type = float(source.training_config["gradient_clipping"]["norm_type"])
    best_loss = initial_loss
    epochs_executed = 0
    for epoch in range(1, OVERFIT_MAX_EPOCHS + 1):
        model.train()
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device_info.device)
            batch_targets = batch_targets.to(device_info.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            optimizer.step()
        epoch_loss, _, _ = _loss_and_error_metrics(
            model,
            subset,
            loss_function=loss_function,
            batch_size=batch_size,
            device=device_info.device,
        )
        best_loss = min(best_loss, epoch_loss)
        epochs_executed = epoch

    final_loss, final_mae, final_rmse = _loss_and_error_metrics(
        model,
        subset,
        loss_function=loss_function,
        batch_size=batch_size,
        device=device_info.device,
    )
    result: dict[str, object] = {
        "diagnostic_id": "E03-D",
        "diagnostic_type": "train_subset_overfit_sanity",
        "source_run_id": str(source.manifest["run_id"]),
        "data_scope": "eligible_train_only",
        "selection": "first_512_chronological_eligible_train_samples",
        "sample_count": len(subset.targets),
        "first_decision_time": subset.decision_times[0].isoformat(),
        "last_decision_time": subset.decision_times[-1].isoformat(),
        "feature_scaler": "restored_train_only_e03_scaler",
        "initial_huber_loss": initial_loss,
        "final_huber_loss": final_loss,
        "best_huber_loss": best_loss,
        "initial_mae": initial_mae,
        "final_mae": final_mae,
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "epochs_executed": epochs_executed,
        "maximum_epochs": OVERFIT_MAX_EPOCHS,
        "device": str(device_info.device),
        "gpu_name": device_info.gpu_name,
        "seed": seed,
        "validation_set": "NOT USED",
        "test_set": "NOT EVALUATED",
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03D_lstm_overfit_{timestamp}",
        result=result,
    )
