from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from btc_forecasting.baselines.naive import zero_return_prediction
from btc_forecasting.baselines.ridge import (
    RidgeSamples,
    build_lookback_samples,
    load_development_samples,
)
from btc_forecasting.common.config import load_yaml
from btc_forecasting.common.hashing import sha256_file
from btc_forecasting.common.seed import seed_everything
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.evaluation.metrics import regression_loss_metrics, regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH

F0_CONFIG_RELATIVE_PATH = Path("configs/features/f0_minimal.yaml")
LSTM_CONFIG_RELATIVE_PATH = Path("configs/models/lstm.yaml")
TRAINING_CONFIG_RELATIVE_PATH = Path("configs/training.yaml")
E03_CONFIG_RELATIVE_PATH = Path("configs/experiments/e03.yaml")
LOOKBACK_HOURS = 24


@dataclass(frozen=True)
class SequenceSamples:
    features: np.ndarray
    targets: np.ndarray
    decision_times: pd.DatetimeIndex
    candidate_count: int
    excluded_lookback_count: int


@dataclass(frozen=True)
class PreparedSequenceSamples:
    train: SequenceSamples
    validation: SequenceSamples


@dataclass(frozen=True)
class ScaledSequenceSamples:
    train: SequenceSamples
    validation: SequenceSamples
    feature_scaler: RobustScaler


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    gpu_name: str


@dataclass(frozen=True)
class TrainingOutcome:
    model: LSTMRegressor
    history: list[dict[str, float | int]]
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float
    duration_seconds: float


@dataclass(frozen=True)
class LSTMRunResult:
    metrics_path: Path
    run_directory: Path
    result: dict[str, object]


def _sequence_view(samples: RidgeSamples) -> SequenceSamples:
    expected_width = LOOKBACK_HOURS * len(F0_FEATURE_NAMES)
    if samples.features.ndim != 2 or samples.features.shape[1] != expected_width:
        raise ValueError(f"Expected flattened windows with width {expected_width}")
    features = samples.features.reshape(-1, LOOKBACK_HOURS, len(F0_FEATURE_NAMES))
    return SequenceSamples(
        features=features.astype(np.float32, copy=False),
        targets=samples.targets.astype(np.float64, copy=False),
        decision_times=samples.decision_times,
        candidate_count=samples.candidate_count,
        excluded_lookback_count=samples.excluded_lookback_count,
    )


def build_sequence_samples(
    feature_rows: pd.DataFrame,
    targets: pd.DataFrame,
) -> SequenceSamples:
    """Build E03 sequences by reusing the frozen gap-safe window logic."""
    return _sequence_view(build_lookback_samples(feature_rows, targets))


def load_sequence_development_samples(
    *,
    canonical_path: Path,
    target_path: Path,
    split_metadata_path: Path,
) -> PreparedSequenceSamples:
    """Load train and validation only, with the final test partition filtered at read time."""
    prepared = load_development_samples(
        canonical_path=canonical_path,
        target_path=target_path,
        split_metadata_path=split_metadata_path,
    )
    return PreparedSequenceSamples(
        train=_sequence_view(prepared.train),
        validation=_sequence_view(prepared.validation),
    )


def fit_train_feature_scaler(
    train: SequenceSamples,
    validation: SequenceSamples,
) -> ScaledSequenceSamples:
    """Fit one scaler per F0 feature from train timesteps and apply it to validation."""
    if len(train.targets) == 0:
        raise ValueError("LSTM training samples must not be empty")
    feature_count = len(F0_FEATURE_NAMES)
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
    )
    train_2d = train.features.reshape(-1, feature_count)
    validation_2d = validation.features.reshape(-1, feature_count)
    scaled_train = scaler.fit_transform(train_2d).reshape(train.features.shape)
    scaled_validation = scaler.transform(validation_2d).reshape(validation.features.shape)

    def replace_features(samples: SequenceSamples, features: np.ndarray) -> SequenceSamples:
        return SequenceSamples(
            features=features.astype(np.float32, copy=False),
            targets=samples.targets,
            decision_times=samples.decision_times,
            candidate_count=samples.candidate_count,
            excluded_lookback_count=samples.excluded_lookback_count,
        )

    return ScaledSequenceSamples(
        train=replace_features(train, scaled_train),
        validation=replace_features(validation, scaled_validation),
        feature_scaler=scaler,
    )


def require_official_cuda() -> DeviceInfo:
    """Return the official device or fail without a CPU fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "E03 official training requires CUDA, but torch.cuda.is_available() is false"
        )
    return DeviceInfo(device=torch.device("cuda"), gpu_name=torch.cuda.get_device_name(0))


def configure_determinism(seed: int) -> None:
    seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_loader(
    samples: SequenceSamples,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(samples.features),
        torch.from_numpy(samples.targets.astype(np.float32, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def build_training_components(
    model: LSTMRegressor,
    *,
    training_config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, nn.HuberLoss, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer_config = training_config["optimizer"]
    loss_config = training_config["loss"]
    scheduler_config = training_config["scheduler"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    loss = nn.HuberLoss(
        delta=float(loss_config["delta"]),
        reduction=str(loss_config["reduction"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["max_epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    return optimizer, loss, scheduler


def fit_lstm(
    model: LSTMRegressor,
    *,
    train: SequenceSamples,
    validation: SequenceSamples,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> TrainingOutcome:
    batch_size = int(training_config["batch_size"])
    train_loader = _make_loader(
        train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    model.to(device)
    optimizer, loss_function, scheduler = build_training_components(
        model,
        training_config=training_config,
    )
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping"]["patience"])
    max_norm = float(training_config["gradient_clipping"]["max_norm"])
    norm_type = float(training_config["gradient_clipping"]["norm_type"])
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            optimizer.step()
            train_loss_sum += float(loss.detach().item()) * len(batch_targets)
            train_count += len(batch_targets)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                loss = loss_function(model(batch_features), batch_targets)
                validation_loss_sum += float(loss.item()) * len(batch_targets)
                validation_count += len(batch_targets)

        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_huber_loss": train_loss,
                "validation_huber_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        scheduler.step()
        if bad_epochs >= patience:
            break

    duration = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("E03 training completed without a best checkpoint")
    model.load_state_dict(best_state)
    return TrainingOutcome(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_trained=len(history),
        best_validation_loss=best_loss,
        duration_seconds=duration,
    )


def predict_lstm(
    model: LSTMRegressor,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def _sample_identity(decision_times: pd.DatetimeIndex) -> str:
    values = decision_times.asi8.astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def evaluate_validation(
    *,
    validation: SequenceSamples,
    predictions: np.ndarray,
    train: SequenceSamples,
) -> dict[str, object]:
    """Evaluate validation and its same-row zero baseline; no test input is accepted."""
    metrics = regression_metrics(validation.targets, predictions)
    zero_predictions = zero_return_prediction(len(validation.targets))
    zero_metrics = regression_loss_metrics(validation.targets, zero_predictions)
    identity = _sample_identity(validation.decision_times)
    zero_mae = float(zero_metrics["mae"])
    zero_rmse = float(zero_metrics["rmse"])
    return {
        "experiment_id": "E03",
        "model": "lstm",
        "evaluated_splits": ["validation"],
        "data": {
            "lookback_hours": LOOKBACK_HOURS,
            "window_shape": [LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
            "train_candidate_rows": train.candidate_count,
            "train_eligible_samples": len(train.targets),
            "train_excluded_invalid_lookback": train.excluded_lookback_count,
            "validation_candidate_rows": validation.candidate_count,
            "validation_eligible_samples": len(validation.targets),
            "validation_excluded_invalid_lookback": validation.excluded_lookback_count,
            "total_excluded_invalid_lookback": (
                train.excluded_lookback_count + validation.excluded_lookback_count
            ),
        },
        "lstm_validation": {
            "n": int(metrics["n"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "pearson_ic": float(metrics["pearson_ic"]),
            "spearman_rank_ic": float(metrics["spearman_rank_ic"]),
            "directional_accuracy": float(metrics["directional_accuracy"]),
            "sample_identity_sha256": identity,
        },
        "zero_return_same_validation_rows": {
            "n": int(zero_metrics["n"]),
            "mae": zero_mae,
            "rmse": zero_rmse,
            "sample_identity_sha256": identity,
        },
        "skill": {
            "mae": 1.0 - float(metrics["mae"]) / zero_mae,
            "rmse": 1.0 - float(metrics["rmse"]) / zero_rmse,
        },
        "test_set": "NOT EVALUATED",
    }


def _load_resolved_configuration(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    feature = load_yaml(root / F0_CONFIG_RELATIVE_PATH)["feature_set"]
    model = load_yaml(root / LSTM_CONFIG_RELATIVE_PATH)["model"]
    training = load_yaml(root / TRAINING_CONFIG_RELATIVE_PATH)["training"]
    experiment = load_yaml(root / E03_CONFIG_RELATIVE_PATH)["experiment"]
    if tuple(feature["features"]) != F0_FEATURE_NAMES:
        raise ValueError("Frozen F0 config does not match the implemented feature order")
    if experiment.get("training_protocol") != "deep_training_v1":
        raise ValueError("E03 must use Deep Training Protocol v1")
    if experiment.get("lookback") != "24h" or int(experiment["seed"]) != 42:
        raise ValueError("E03 frozen lookback and seed do not match the implementation")
    if training.get("protocol_version") != "deep_training_v1":
        raise ValueError("Training config must declare Deep Training Protocol v1")
    if training["device"] != {
        "official": "cuda",
        "require_available": True,
        "run_metadata": ["actual_device", "gpu_name"],
    }:
        raise ValueError("E03 official device configuration must require CUDA")
    return feature, model, training, experiment


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def run_lstm_baseline(*, project_root: Path) -> LSTMRunResult:
    """Execute the official CUDA-only E03 development experiment."""
    root = project_root.resolve()
    feature_config, model_config, training_config, experiment_config = (
        _load_resolved_configuration(root)
    )
    device_info = require_official_cuda()
    seed = int(experiment_config["seed"])
    configure_determinism(seed)

    canonical_path = root / CANONICAL_1H_RELATIVE_PATH
    target_path = root / TARGET_RELATIVE_PATH
    split_path = root / FROZEN_SPLIT_RELATIVE_PATH
    prepared = load_sequence_development_samples(
        canonical_path=canonical_path,
        target_path=target_path,
        split_metadata_path=split_path,
    )
    scaled = fit_train_feature_scaler(prepared.train, prepared.validation)
    model = LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        configured_dropout=float(model_config["dropout"]),
    )
    outcome = fit_lstm(
        model,
        train=scaled.train,
        validation=scaled.validation,
        training_config=training_config,
        seed=seed,
        device=device_info.device,
    )
    validation_predictions = predict_lstm(
        outcome.model,
        scaled.validation.features,
        batch_size=int(training_config["batch_size"]),
        device=device_info.device,
    )
    result = evaluate_validation(
        validation=scaled.validation,
        predictions=validation_predictions,
        train=scaled.train,
    )
    result["training"] = {
        "device": str(device_info.device),
        "gpu_name": device_info.gpu_name,
        "seed": seed,
        "epochs_trained": outcome.epochs_trained,
        "best_epoch": outcome.best_epoch,
        "best_validation_huber_loss": outcome.best_validation_loss,
        "duration_seconds": outcome.duration_seconds,
        "configured_dropout": model.configured_dropout,
        "effective_lstm_dropout": model.effective_lstm_dropout,
    }

    run_id = f"E03_1h_F0_B2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = root / "outputs" / "runs" / run_id
    model_directory = root / "artifacts" / "models" / run_id
    scaler_directory = root / "artifacts" / "scalers" / run_id
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.parquet"
    history_path = run_directory / "training_history.json"
    resolved_config_path = run_directory / "resolved_config.json"
    checkpoint_path = model_directory / "lstm.pt"
    scaler_path = scaler_directory / "feature_scaler.joblib"

    manifest = create_run_manifest(project_root=root, experiment_id="E03", run_id=run_id)
    manifest.update(
        {
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "seed": seed,
            "configured_dropout": model.configured_dropout,
            "effective_lstm_dropout": model.effective_lstm_dropout,
            "model_config": model_config,
            "training_config": training_config,
            "feature_set": feature_config["id"],
            "feature_names": list(F0_FEATURE_NAMES),
            "input_shape": ["batch", LOOKBACK_HOURS, len(F0_FEATURE_NAMES)],
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
            "experiment": experiment_config,
            "feature_set": feature_config,
            "model": model_config,
            "training": training_config,
        },
    )
    prediction_frame = pd.DataFrame(
        {
            "decision_time": scaled.validation.decision_times,
            TARGET_COLUMN: scaled.validation.targets,
            "lstm_prediction": validation_predictions,
            "zero_return_prediction": zero_return_prediction(len(validation_predictions)),
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
            "model_config": model_config,
            "training_config": training_config,
            "feature_names": list(F0_FEATURE_NAMES),
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
