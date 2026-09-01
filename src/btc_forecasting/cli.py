from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from btc_forecasting.common.config import find_project_root, list_yaml_files, load_yaml
from btc_forecasting.data.acquisition import run_acquisition
from btc_forecasting.data.acquisition_config import load_acquisition_config
from btc_forecasting.baselines.ridge import run_ridge_baseline
from btc_forecasting.baselines.zero_return import run_zero_return_baseline
from btc_forecasting.data.canonical_1h import run_canonical_1h_build
from btc_forecasting.data.canonical_5m import run_canonical_5m_build
from btc_forecasting.data.raw_validation import run_raw_validation
from btc_forecasting.experiments.registry import load_experiment_registry
from btc_forecasting.reporting.manifest import create_run_manifest, write_manifest
from btc_forecasting.splits.frozen import run_frozen_chronological_split
from btc_forecasting.targets.one_hour import run_one_hour_target_build


def _doctor(root: Path) -> int:
    print(f"project_root={root}")
    print(f"python={sys.version.split()[0]}")
    print(f"python_supported={sys.version_info[:2] == (3, 13)}")
    print(f"uv={shutil.which('uv') or 'NOT_FOUND'}")
    print(f"config_files={len(list_yaml_files(root))}")

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        print("torch=NOT_INSTALLED (expected for base environment)")
    else:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")
    return 0


def _validate_configs(root: Path) -> int:
    errors: list[str] = []
    for path in list_yaml_files(root):
        try:
            load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(root)}: {exc}")

    try:
        registry = load_experiment_registry(root)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"experiment registry: {exc}")
        registry = {}

    acquisition_config_path = root / "configs" / "data_acquisition.yaml"
    if acquisition_config_path.is_file():
        try:
            load_acquisition_config(acquisition_config_path, project_root=root)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"data acquisition config: {exc}")

    feature_ids: set[str] = set()
    for path in sorted((root / "configs" / "features").glob("*.yaml")):
        document = load_yaml(path)
        feature = document.get("feature_set", {})
        feature_id = feature.get("id") if isinstance(feature, dict) else None
        if isinstance(feature_id, str):
            feature_ids.add(feature_id)

    model_ids: set[str] = set()
    for path in sorted((root / "configs" / "models").glob("*.yaml")):
        document = load_yaml(path)
        model = document.get("model", {})
        model_id = model.get("id") if isinstance(model, dict) else None
        if isinstance(model_id, str):
            model_ids.add(model_id)

    for experiment_id, experiment in registry.items():
        feature_set = experiment.get("feature_set")
        if isinstance(feature_set, str) and feature_set in {"F0", "F1", "F2"}:
            if feature_set not in feature_ids:
                errors.append(f"{experiment_id}: unknown feature set {feature_set}")
        model = experiment.get("model")
        if isinstance(model, str) and model.startswith("B") and model not in model_ids:
            errors.append(f"{experiment_id}: unknown model {model}")

    if errors:
        print("CONFIG VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"CONFIG VALIDATION PASSED: {len(list_yaml_files(root))} YAML files, "
        f"{len(registry)} experiments, {len(feature_ids)} feature sets, {len(model_ids)} models"
    )
    return 0


def _show_plan(root: Path) -> int:
    registry = load_experiment_registry(root)
    for experiment_id in sorted(registry):
        experiment = registry[experiment_id]
        print(
            f"{experiment_id}: {experiment.get('name')} | "
            f"status={experiment.get('status')} | stage={experiment.get('stage')}"
        )
    return 0


def _init_run(root: Path, experiment_id: str) -> int:
    registry = load_experiment_registry(root)
    if experiment_id not in registry:
        raise ValueError(f"Unknown experiment ID: {experiment_id}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_id}_{timestamp}"
    run_dir = root / "outputs" / "runs" / run_id

    data_config = load_yaml(root / "configs" / "data.yaml")
    raw_path = root / data_config["data"]["raw_master"]["path"]
    manifest = create_run_manifest(
        project_root=root,
        experiment_id=experiment_id,
        run_id=run_id,
        raw_data_path=raw_path,
    )
    manifest["experiment"] = registry[experiment_id]
    write_manifest(run_dir / "manifest.json", manifest)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"experiment": registry[experiment_id]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(run_dir)
    return 0


def _acquire_data(
    root: Path,
    config_argument: Path,
    *,
    dry_run: bool,
    verify_only: bool,
) -> int:
    config_path = config_argument if config_argument.is_absolute() else root / config_argument
    config = load_acquisition_config(config_path, project_root=root)
    result = run_acquisition(
        project_root=root,
        config_path=config_path,
        config=config,
        dry_run=dry_run,
        verify_only=verify_only,
    )
    if dry_run:
        print(json.dumps(result.plan, indent=2, sort_keys=True))
    else:
        if result.summary is not None:
            print(json.dumps(result.summary, indent=2, sort_keys=True))
        if result.manifest_path is not None:
            print(f"manifest={result.manifest_path.relative_to(root)}")
        if result.summary_path is not None:
            print(f"summary={result.summary_path.relative_to(root)}")
    return result.exit_code


def _validate_raw_data(root: Path, config_argument: Path) -> int:
    config_path = config_argument if config_argument.is_absolute() else root / config_argument
    config = load_acquisition_config(config_path, project_root=root)
    result = run_raw_validation(
        project_root=root,
        config_path=config_path,
        config=config,
    )
    display_summary = dict(result.summary)
    display_summary.pop("ignore_value_counts", None)
    print(json.dumps(display_summary, indent=2, sort_keys=True))
    print(f"summary={result.summary_path.relative_to(root)}")
    print(f"archive_validation={result.archive_validation_path.relative_to(root)}")
    print(f"gaps={result.gaps_path.relative_to(root)}")
    print(f"timestamp_anomalies={result.timestamp_anomalies_path.relative_to(root)}")
    return result.exit_code


def _build_canonical_5m(root: Path, config_argument: Path) -> int:
    config_path = config_argument if config_argument.is_absolute() else root / config_argument
    config = load_acquisition_config(config_path, project_root=root)
    result = run_canonical_5m_build(project_root=root, config=config)
    print(json.dumps(result.verification, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _build_canonical_1h(root: Path) -> int:
    result = run_canonical_1h_build(project_root=root)
    print(json.dumps(result.verification, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    print(f"incomplete_hours={result.incomplete_hours_path.relative_to(root)}")
    return 0


def _build_eng52_features(root: Path) -> int:
    from btc_forecasting.features.eng52_build import run_eng52_build

    result = run_eng52_build(project_root=root)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    print(f"summary={result.summary_path.relative_to(root)}")
    return 0


def _run_eng52_qc(root: Path) -> int:
    from btc_forecasting.features.eng52_qc import run_eng52_qc

    result = run_eng52_qc(project_root=root)
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _run_eng52_signal_screening(root: Path) -> int:
    from btc_forecasting.features.eng52_screening import run_eng52_signal_screening

    result = run_eng52_signal_screening(project_root=root)
    print(f"screening={result.screening_path.relative_to(root)}")
    print(f"config={result.config_path.relative_to(root)}")
    return 0


def _build_one_hour_targets(root: Path) -> int:
    result = run_one_hour_target_build(project_root=root)
    print(json.dumps(result.verification, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _build_frozen_split(root: Path) -> int:
    result = run_frozen_chronological_split(project_root=root)
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    print(f"metadata={result.metadata_path.relative_to(root)}")
    return 0


def _run_zero_return_baseline(root: Path) -> int:
    result = run_zero_return_baseline(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.metrics_path.relative_to(root)}")
    return 0


def _run_ridge_baseline(root: Path) -> int:
    result = run_ridge_baseline(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.metrics_path.relative_to(root)}")
    return 0


def _run_vn_ridge_control(root: Path) -> int:
    from btc_forecasting.baselines.ridge_vn import run_vn_ridge_control

    result = run_vn_ridge_control(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.metrics_path.parent.relative_to(root)}")
    return 0


def _run_lstm_baseline(root: Path) -> int:
    from btc_forecasting.training.lstm import run_lstm_baseline

    result = run_lstm_baseline(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.run_directory.relative_to(root)}")
    return 0


def _run_lstm_mse_ablation(root: Path) -> int:
    from btc_forecasting.training.lstm_mse import run_lstm_mse_ablation

    result = run_lstm_mse_ablation(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.run_directory.relative_to(root)}")
    return 0


def _run_lstm_vn_mse_experiment(root: Path) -> int:
    from btc_forecasting.training.lstm_vn_mse import run_lstm_vn_mse_experiment

    result = run_lstm_vn_mse_experiment(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.run_directory.relative_to(root)}")
    return 0


def _run_lstm_vn_returns_only(root: Path) -> int:
    from btc_forecasting.training.lstm_vn_returns_only import (
        run_lstm_vn_returns_only,
    )

    result = run_lstm_vn_returns_only(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.run_directory.relative_to(root)}")
    return 0


def _diagnose_lstm_distribution(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.lstm_diagnostics import (
        run_prediction_distribution_diagnostic,
    )

    result = run_prediction_distribution_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _diagnose_lstm_overfit(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.lstm_diagnostics import run_overfit_sanity_diagnostic

    result = run_overfit_sanity_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _diagnose_lstm_generalization(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.lstm_generalization import (
        run_lstm_generalization_diagnostic,
    )

    result = run_lstm_generalization_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _diagnose_volatility_normalization(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.volatility_normalization import (
        run_volatility_normalization_diagnostic,
    )

    result = run_volatility_normalization_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _diagnose_vn_learning(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.lstm_vn_learning_diagnostic import (
        run_vn_learning_diagnostic,
    )

    result = run_vn_learning_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def _diagnose_vn_temporal(root: Path, source_run: Path) -> int:
    from btc_forecasting.training.lstm_vn_temporal_diagnostic import (
        run_vn_temporal_diagnostic,
    )

    result = run_vn_temporal_diagnostic(
        project_root=root,
        source_run=source_run,
    )
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("validate-configs")
    subparsers.add_parser("show-plan")
    init_run = subparsers.add_parser("init-run")
    init_run.add_argument("--experiment", required=True)
    acquire_data = subparsers.add_parser("acquire-data")
    acquire_data.add_argument("--config", type=Path, required=True)
    mode = acquire_data.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    validate_raw_data = subparsers.add_parser("validate-raw-data")
    validate_raw_data.add_argument("--config", type=Path, required=True)
    build_canonical_5m = subparsers.add_parser("build-canonical-5m")
    build_canonical_5m.add_argument("--config", type=Path, required=True)
    subparsers.add_parser("build-canonical-1h")
    subparsers.add_parser("build-eng52-features")
    subparsers.add_parser("run-eng52-qc")
    subparsers.add_parser("run-eng52-signal-screening")
    subparsers.add_parser("build-one-hour-targets")
    subparsers.add_parser("build-frozen-split")
    subparsers.add_parser("run-zero-return-baseline")
    subparsers.add_parser("run-ridge-baseline")
    subparsers.add_parser("run-vn-ridge-control")
    subparsers.add_parser("run-lstm-baseline")
    subparsers.add_parser("run-lstm-mse-ablation")
    subparsers.add_parser("run-lstm-volatility-normalized-mse")
    subparsers.add_parser("run-lstm-vn-returns-only")
    diagnose_distribution = subparsers.add_parser("diagnose-lstm-distribution")
    diagnose_distribution.add_argument("--source-run", type=Path, required=True)
    diagnose_overfit = subparsers.add_parser("diagnose-lstm-overfit")
    diagnose_overfit.add_argument("--source-run", type=Path, required=True)
    diagnose_generalization = subparsers.add_parser("diagnose-lstm-generalization")
    diagnose_generalization.add_argument("--source-run", type=Path, required=True)
    diagnose_volatility = subparsers.add_parser(
        "diagnose-lstm-volatility-normalization"
    )
    diagnose_volatility.add_argument("--source-run", type=Path, required=True)
    diagnose_vn_learning = subparsers.add_parser(
        "diagnose-lstm-vn-learning"
    )
    diagnose_vn_learning.add_argument("--source-run", type=Path, required=True)
    diagnose_vn_temporal = subparsers.add_parser(
        "diagnose-lstm-vn-temporal"
    )
    diagnose_vn_temporal.add_argument("--source-run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_project_root()

    commands: dict[str, Any] = {
        "doctor": lambda: _doctor(root),
        "validate-configs": lambda: _validate_configs(root),
        "show-plan": lambda: _show_plan(root),
        "init-run": lambda: _init_run(root, args.experiment),
        "acquire-data": lambda: _acquire_data(
            root,
            args.config,
            dry_run=args.dry_run,
            verify_only=args.verify_only,
        ),
        "validate-raw-data": lambda: _validate_raw_data(root, args.config),
        "build-canonical-5m": lambda: _build_canonical_5m(root, args.config),
        "build-canonical-1h": lambda: _build_canonical_1h(root),
        "build-eng52-features": lambda: _build_eng52_features(root),
        "run-eng52-qc": lambda: _run_eng52_qc(root),
        "run-eng52-signal-screening": lambda: _run_eng52_signal_screening(root),
        "build-one-hour-targets": lambda: _build_one_hour_targets(root),
        "build-frozen-split": lambda: _build_frozen_split(root),
        "run-zero-return-baseline": lambda: _run_zero_return_baseline(root),
        "run-ridge-baseline": lambda: _run_ridge_baseline(root),
        "run-vn-ridge-control": lambda: _run_vn_ridge_control(root),
        "run-lstm-baseline": lambda: _run_lstm_baseline(root),
        "run-lstm-mse-ablation": lambda: _run_lstm_mse_ablation(root),
        "run-lstm-volatility-normalized-mse": lambda: (
            _run_lstm_vn_mse_experiment(root)
        ),
        "run-lstm-vn-returns-only": lambda: _run_lstm_vn_returns_only(root),
        "diagnose-lstm-distribution": lambda: _diagnose_lstm_distribution(
            root, args.source_run
        ),
        "diagnose-lstm-overfit": lambda: _diagnose_lstm_overfit(root, args.source_run),
        "diagnose-lstm-generalization": lambda: _diagnose_lstm_generalization(
            root, args.source_run
        ),
        "diagnose-lstm-volatility-normalization": lambda: (
            _diagnose_volatility_normalization(root, args.source_run)
        ),
        "diagnose-lstm-vn-learning": lambda: _diagnose_vn_learning(
            root, args.source_run
        ),
        "diagnose-lstm-vn-temporal": lambda: _diagnose_vn_temporal(
            root, args.source_run
        ),
    }
    raise SystemExit(commands[args.command]())
