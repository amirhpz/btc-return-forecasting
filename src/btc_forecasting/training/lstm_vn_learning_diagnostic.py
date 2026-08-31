from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from btc_forecasting.evaluation.metrics import regression_metrics
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.training.lstm import (
    SequenceSamples,
    build_training_components,
    configure_determinism,
    load_sequence_development_samples,
    predict_lstm,
    require_official_cuda,
)
from btc_forecasting.training.lstm_diagnostics import (
    DiagnosticRunResult,
    OVERFIT_MAX_EPOCHS,
    SourceRun,
    _finite_float,
    _read_json,
    _resolve_inside,
    _restore_model,
    _safe_ratio,
    _write_diagnostic,
    apply_restored_scaler,
    select_overfit_subset,
)
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    prepare_volatility_normalized_samples,
    reconstruct_raw_predictions,
    resolve_vn_mse_configuration,
)

DIAGNOSTIC_ID = "E03-VN-LD"


def validate_vn_mse_source_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("experiment_id") != "E03-VN-MSE":
        raise ValueError("E03-VN-LD requires an E03-VN-MSE source run")
    if manifest.get("test_set") != "NOT EVALUATED":
        raise ValueError("E03-VN-LD refuses a source run that accessed the test set")
    if manifest.get("evaluated_splits") != ["validation"]:
        raise ValueError("E03-VN-LD requires a validation-only source run")
    difference = manifest.get("controlled_difference")
    if not isinstance(difference, dict) or difference.get("epsilon_added") is not False:
        raise ValueError("E03-VN-LD requires the no-epsilon normalized target")


def load_vn_mse_source_run(*, project_root: Path, source_run: Path) -> SourceRun:
    root = project_root.resolve()
    run_directory = _resolve_inside(root, source_run)
    manifest = _read_json(run_directory / "manifest.json")
    validate_vn_mse_source_manifest(manifest)
    resolved = resolve_vn_mse_configuration(project_root=root)
    model_config = manifest["model_config"]
    training_config = manifest["training_config"]
    if model_config != resolved.model:
        raise ValueError("Current frozen LSTM config differs from the E03-VN-MSE run")
    if training_config != resolved.training:
        raise ValueError("Current E03-VN-MSE training semantics differ from the source run")
    artifacts = manifest["artifacts"]
    data = manifest["data"]
    source = SourceRun(
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
        source.checkpoint_path,
        source.scaler_path,
        source.canonical_path,
        source.target_path,
        source.split_path,
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing E03-VN-LD source artifacts: {missing}")
    return source


def normalized_target_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Normalized targets and predictions must have equal non-empty shape")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Normalized targets and predictions must be finite")
    metrics = regression_metrics(truth, predicted)
    mse = float(np.mean(np.square(truth - predicted)))
    mae = float(np.mean(np.abs(truth - predicted)))
    target_std = float(np.std(truth, ddof=0))
    prediction_std = float(np.std(predicted, ddof=0))
    zero_mse = float(np.mean(np.square(truth)))
    zero_mae = float(np.mean(np.abs(truth)))
    return {
        "n": int(len(truth)),
        "mse": mse,
        "mae": mae,
        "pearson_ic": _finite_float(metrics["pearson_ic"]),
        "spearman_rank_ic": _finite_float(metrics["spearman_rank_ic"]),
        "directional_accuracy": _finite_float(metrics["directional_accuracy"]),
        "target_mean": float(np.mean(truth)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": prediction_std,
        "prediction_std_over_target_std": _safe_ratio(prediction_std, target_std),
        "zero_same_rows": {
            "mse": zero_mse,
            "mae": zero_mae,
        },
        "skill": {
            "mse": None if zero_mse == 0.0 else 1.0 - mse / zero_mse,
            "mae": None if zero_mae == 0.0 else 1.0 - mae / zero_mae,
        },
    }


def raw_return_metrics(
    targets: np.ndarray,
    normalized_predictions: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(targets, dtype=np.float64)
    predictions = reconstruct_raw_predictions(normalized_predictions, sigma)
    if truth.shape != predictions.shape:
        raise ValueError("Raw targets and reconstructed predictions must have equal shape")
    metrics = regression_metrics(truth, predictions)
    return {
        "n": int(metrics["n"]),
        "mae": _finite_float(metrics["mae"]),
        "rmse": _finite_float(metrics["rmse"]),
        "r2": _finite_float(metrics["r2"]),
        "pearson_ic": _finite_float(metrics["pearson_ic"]),
        "spearman_rank_ic": _finite_float(metrics["spearman_rank_ic"]),
        "directional_accuracy": _finite_float(metrics["directional_accuracy"]),
    }


def restored_checkpoint_split_report(
    samples: VolatilityNormalizedSamples,
    normalized_predictions: np.ndarray,
) -> dict[str, object]:
    return {
        "normalized_target": normalized_target_metrics(
            samples.normalized.targets,
            normalized_predictions,
        ),
        "raw_reconstructed_return": raw_return_metrics(
            samples.raw.targets,
            normalized_predictions,
            samples.sigma,
        ),
        "invalid_sigma_exclusions": samples.exclusion_count,
    }


def build_overfit_summary(
    *,
    subset: SequenceSamples,
    initial_predictions: np.ndarray,
    final_predictions: np.ndarray,
    best_mse: float,
    epochs_executed: int,
) -> dict[str, object]:
    initial = normalized_target_metrics(subset.targets, initial_predictions)
    final = normalized_target_metrics(subset.targets, final_predictions)
    return {
        "data_scope": "eligible_train_only",
        "selection": "first_512_chronological_eligible_e03_vn_mse_train_samples",
        "sample_count": int(len(subset.targets)),
        "first_decision_time": subset.decision_times[0].isoformat(),
        "last_decision_time": subset.decision_times[-1].isoformat(),
        "initial_mse": initial["mse"],
        "final_mse": final["mse"],
        "best_mse": float(best_mse),
        "initial_mae": initial["mae"],
        "final_mae": final["mae"],
        "target_std": final["target_std"],
        "final_prediction_std": final["prediction_std"],
        "final_pearson_ic": final["pearson_ic"],
        "epochs_executed": int(epochs_executed),
        "maximum_epochs": OVERFIT_MAX_EPOCHS,
        "validation_set": "NOT USED",
        "test_set": "NOT EVALUATED",
    }


def build_learning_diagnostic_report(
    *,
    source_run_id: str,
    train_report: dict[str, object],
    validation_report: dict[str, object],
    overfit_report: dict[str, object],
    device: str,
    gpu_name: str,
    seed: int,
) -> dict[str, object]:
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "diagnostic_type": "normalized_target_learning",
        "source_run_id": source_run_id,
        "device": device,
        "gpu_name": gpu_name,
        "seed": seed,
        "restored_checkpoint": {
            "evaluated_splits": ["train", "validation"],
            "train": train_report,
            "validation": validation_report,
        },
        "normalized_target_overfit_sanity": overfit_report,
        "test_set": "NOT EVALUATED",
    }


def _predict_mse(
    model: LSTMRegressor,
    samples: SequenceSamples,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    predictions = predict_lstm(
        model,
        samples.features,
        batch_size=batch_size,
        device=device,
    )
    mse = float(np.mean(np.square(samples.targets - predictions)))
    return predictions, mse


def _run_normalized_overfit_sanity(
    *,
    train: SequenceSamples,
    source: SourceRun,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    subset = select_overfit_subset(train)
    model = LSTMRegressor(
        input_size=len(F0_FEATURE_NAMES),
        hidden_size=int(source.model_config["hidden_size"]),
        num_layers=int(source.model_config["num_layers"]),
        configured_dropout=float(source.model_config["dropout"]),
    ).to(device)
    optimizer, loss_function, _ = build_training_components(
        model,
        training_config=source.training_config,
    )
    if not isinstance(loss_function, torch.nn.MSELoss):
        raise TypeError("E03-VN-LD overfit sanity requires MSELoss")
    batch_size = int(source.training_config["batch_size"])
    initial_predictions, initial_mse = _predict_mse(
        model,
        subset,
        batch_size=batch_size,
        device=device,
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
    best_mse = initial_mse
    epochs_executed = 0
    for epoch in range(1, OVERFIT_MAX_EPOCHS + 1):
        model.train()
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
        _, epoch_mse = _predict_mse(
            model,
            subset,
            batch_size=batch_size,
            device=device,
        )
        best_mse = min(best_mse, epoch_mse)
        epochs_executed = epoch
    final_predictions, _ = _predict_mse(
        model,
        subset,
        batch_size=batch_size,
        device=device,
    )
    return build_overfit_summary(
        subset=subset,
        initial_predictions=initial_predictions,
        final_predictions=final_predictions,
        best_mse=best_mse,
        epochs_executed=epochs_executed,
    )


def run_vn_learning_diagnostic(
    *,
    project_root: Path,
    source_run: Path,
) -> DiagnosticRunResult:
    root = project_root.resolve()
    source = load_vn_mse_source_run(project_root=root, source_run=source_run)
    device_info = require_official_cuda()
    seed = int(source.manifest["seed"])
    configure_determinism(seed)
    prepared = load_sequence_development_samples(
        canonical_path=source.canonical_path,
        target_path=source.target_path,
        split_metadata_path=source.split_path,
    )
    train = prepare_volatility_normalized_samples(prepared.train)
    validation = prepare_volatility_normalized_samples(prepared.validation)
    scaler = joblib.load(source.scaler_path)
    if not isinstance(scaler, RobustScaler):
        raise TypeError("E03-VN-LD source scaler is not a RobustScaler")
    scaled_train = apply_restored_scaler(train.normalized, scaler)
    scaled_validation = apply_restored_scaler(validation.normalized, scaler)
    restored_model = _restore_model(source, device=device_info.device)
    batch_size = int(source.training_config["batch_size"])
    train_predictions = predict_lstm(
        restored_model,
        scaled_train.features,
        batch_size=batch_size,
        device=device_info.device,
    )
    validation_predictions = predict_lstm(
        restored_model,
        scaled_validation.features,
        batch_size=batch_size,
        device=device_info.device,
    )
    train_report = restored_checkpoint_split_report(train, train_predictions)
    validation_report = restored_checkpoint_split_report(
        validation,
        validation_predictions,
    )
    configure_determinism(seed)
    overfit_report = _run_normalized_overfit_sanity(
        train=scaled_train,
        source=source,
        device=device_info.device,
        seed=seed,
    )
    result = build_learning_diagnostic_report(
        source_run_id=str(source.manifest["run_id"]),
        train_report=train_report,
        validation_report=validation_report,
        overfit_report=overfit_report,
        device=str(device_info.device),
        gpu_name=device_info.gpu_name,
        seed=seed,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _write_diagnostic(
        root=root,
        directory_name=f"E03VNLD_learning_{timestamp}",
        result=result,
    )
