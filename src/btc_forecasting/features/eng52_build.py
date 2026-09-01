from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecasting.common.config import load_yaml
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES, compute_eng52_features

ENG52_AUDIT_RELATIVE_PATH = Path("configs/features/eng52_audit.yaml")
ENG52_OUTPUT_RELATIVE_PATH = Path("data/processed/btcusdt_1h_eng52_v001.parquet")
ENG52_SUMMARY_RELATIVE_PATH = Path("outputs/data/eng52/summary.json")


@dataclass(frozen=True)
class Eng52BuildResult:
    artifact_path: Path
    summary_path: Path
    summary: dict[str, object]


def _configured_feature_names(audit_path: Path) -> tuple[str, ...]:
    document = load_yaml(audit_path)
    entries = document.get("features")
    if not isinstance(entries, list):
        raise ValueError("ENG52 audit must contain a features list")
    names = tuple(
        entry.get("feature_name")
        for entry in entries
        if isinstance(entry, dict)
    )
    if len(names) != 52 or len(set(names)) != 52 or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("ENG52 audit must define exactly 52 unique feature names")
    if names != ENG52_FEATURE_NAMES:
        raise ValueError("ENG52 production registry does not match configured audit order")
    return names  # type: ignore[return-value]


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def build_eng52_dataset(
    *,
    source_path: Path,
    output_path: Path,
    summary_path: Path,
    audit_path: Path,
) -> Eng52BuildResult:
    feature_names = _configured_feature_names(audit_path)
    source_signature_before = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    source = pd.read_parquet(source_path)
    features = compute_eng52_features(source)
    expected_columns = ("open_time", *feature_names)
    if tuple(features.columns) != expected_columns or len(features) != len(source):
        raise ValueError("ENG52 output schema or row count is invalid")

    per_feature: dict[str, dict[str, int | float]] = {}
    for name in feature_names:
        finite_count = int(np.count_nonzero(np.isfinite(features[name].to_numpy(float))))
        missing_count = len(features) - finite_count
        per_feature[name] = {
            "finite_count": finite_count,
            "missing_count": missing_count,
            "missing_ratio": missing_count / len(features),
        }
    summary: dict[str, object] = {
        "input_row_count": len(source),
        "output_row_count": len(features),
        "first_open_time": _timestamp_text(features["open_time"].iloc[0]),
        "last_open_time": _timestamp_text(features["open_time"].iloc[-1]),
        "feature_names": list(feature_names),
        "per_feature": per_feature,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    summary_temporary = summary_path.with_suffix(f"{summary_path.suffix}.tmp")
    features.to_parquet(output_temporary, index=False)
    summary_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_signature_after = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    if source_signature_after != source_signature_before:
        raise ValueError("Canonical 1-hour source was modified during ENG52 construction")
    os.replace(output_temporary, output_path)
    os.replace(summary_temporary, summary_path)
    return Eng52BuildResult(output_path, summary_path, summary)


def run_eng52_build(*, project_root: Path) -> Eng52BuildResult:
    root = project_root.resolve()
    return build_eng52_dataset(
        source_path=root / CANONICAL_1H_RELATIVE_PATH,
        output_path=root / ENG52_OUTPUT_RELATIVE_PATH,
        summary_path=root / ENG52_SUMMARY_RELATIVE_PATH,
        audit_path=root / ENG52_AUDIT_RELATIVE_PATH,
    )
