from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from btc_forecasting.common.config import load_yaml
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES
from btc_forecasting.features.eng52_build import ENG52_OUTPUT_RELATIVE_PATH
from btc_forecasting.features.eng52_qc import QC_SUMMARY_RELATIVE_PATH
from btc_forecasting.features.f0 import compute_f0_features
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm_generalization import _correlation

SCREENING_OUTPUT_RELATIVE_PATH = Path("outputs/data/eng52/signal_screening.json")
STABLE_CONFIG_RELATIVE_PATH = Path("configs/features/eng52_stable_candidates.yaml")
VOLATILITY_FEATURE_NAME = "rolling_volatility_24h"
LOOKBACK_HOURS = 24
BLOCK_COUNT = 6
MIN_SIGN_CONSISTENT_BLOCKS = 5
MIN_ABSOLUTE_SPEARMAN = 0.02
ONE_HOUR = timedelta(hours=1)
TARGET_COLUMNS = ("bar_open_time", "decision_time", "target_time", TARGET_COLUMN)
SPARSE_FEATURE_NAMES = (
    "breakdown_strength_20",
    "breakout_strength_20",
    "mom_tl_break_bull_30",
)
STABLE_CANDIDATE = "STABLE_CANDIDATE"
WEAK_OR_UNSTABLE = "WEAK_OR_UNSTABLE"
REDUNDANT_ALTERNATE = "REDUNDANT_ALTERNATE"


@dataclass(frozen=True)
class ScreeningSamples:
    endpoint_features: pd.DataFrame
    normalized_targets: np.ndarray
    decision_times: pd.DatetimeIndex
    candidate_count: int
    excluded_incomplete_window: int
    excluded_nonfinite_eng52_window: int
    excluded_invalid_sigma: int
    endpoint_missing_ratios: dict[str, float]


@dataclass(frozen=True)
class Eng52ScreeningResult:
    screening_path: Path
    config_path: Path
    report: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _load_train_targets(
    target_path: Path,
    *,
    split_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Physically read only target rows inside the frozen TRAIN decision period."""
    boundaries = split_metadata["split"]["boundaries"]
    train_start = _parse_utc(boundaries["train"]["decision_time_start_inclusive"])
    validation_start = _parse_utc(
        boundaries["validation"]["decision_time_start_inclusive"]
    )
    train_target_end = _parse_utc(boundaries["train"]["target_time_end_exclusive"])
    rows = pq.read_table(
        target_path,
        columns=list(TARGET_COLUMNS),
        filters=[
            ("decision_time", ">=", train_start),
            ("decision_time", "<", validation_start),
        ],
    ).to_pandas()
    decision_time = pd.to_datetime(rows["decision_time"], utc=True)
    target_time = pd.to_datetime(rows["target_time"], utc=True)
    retained = rows.loc[
        (decision_time >= train_start)
        & (decision_time < validation_start)
        & (target_time < train_target_end)
    ].sort_values("decision_time", kind="stable").reset_index(drop=True)
    expected = int(split_metadata["split"]["retained_rows"]["train"])
    if len(retained) != expected:
        raise ValueError(
            "TRAIN target count does not match frozen split metadata: "
            f"expected={expected}, actual={len(retained)}"
        )
    return retained


def build_screening_samples(
    eng52_rows: pd.DataFrame,
    f0_rows: pd.DataFrame,
    train_targets: pd.DataFrame,
) -> ScreeningSamples:
    missing_eng52 = sorted({"open_time", *ENG52_FEATURE_NAMES} - set(eng52_rows.columns))
    if missing_eng52:
        raise ValueError(f"Missing ENG52 columns: {missing_eng52}")
    if VOLATILITY_FEATURE_NAME not in f0_rows:
        raise ValueError(f"Missing F0 volatility feature: {VOLATILITY_FEATURE_NAME}")
    missing_targets = sorted(set(TARGET_COLUMNS) - set(train_targets.columns))
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    eng52_time = pd.DatetimeIndex(pd.to_datetime(eng52_rows["open_time"], utc=True))
    f0_time = pd.DatetimeIndex(pd.to_datetime(f0_rows["open_time"], utc=True))
    if eng52_time.has_duplicates or not eng52_time.is_monotonic_increasing:
        raise ValueError("ENG52 timestamps must be strictly ordered and unique")
    if not eng52_time.equals(f0_time):
        raise ValueError("F0 and ENG52 timestamps must align one-to-one")

    ordered_targets = train_targets.sort_values("decision_time", kind="stable").reset_index(drop=True)
    bar_time = pd.DatetimeIndex(pd.to_datetime(ordered_targets["bar_open_time"], utc=True))
    decision_time = pd.DatetimeIndex(pd.to_datetime(ordered_targets["decision_time"], utc=True))
    if not (decision_time - bar_time == ONE_HOUR).all():
        raise ValueError("Feature anchors must become available exactly one hour later")

    positions = eng52_time.get_indexer(bar_time)
    eng52_values = eng52_rows.loc[:, list(ENG52_FEATURE_NAMES)].to_numpy(np.float64)
    sigma = f0_rows[VOLATILITY_FEATURE_NAME].to_numpy(np.float64)
    targets = ordered_targets[TARGET_COLUMN].to_numpy(np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("Frozen TRAIN targets must be finite")

    endpoint_values = np.full((len(positions), len(ENG52_FEATURE_NAMES)), np.nan)
    present = positions >= 0
    endpoint_values[present] = eng52_values[positions[present]]
    endpoint_missing_ratios = {
        name: float(np.mean(~np.isfinite(endpoint_values[:, index])))
        for index, name in enumerate(ENG52_FEATURE_NAMES)
    }

    eligible_features: list[np.ndarray] = []
    normalized_targets: list[float] = []
    eligible_decisions: list[pd.Timestamp] = []
    incomplete_window = 0
    nonfinite_window = 0
    invalid_sigma = 0
    expected_span = (LOOKBACK_HOURS - 1) * ONE_HOUR
    for target_position, end in enumerate(positions):
        start = end - LOOKBACK_HOURS + 1
        if end < 0 or start < 0 or eng52_time[end] - eng52_time[start] != expected_span:
            incomplete_window += 1
            continue
        window = eng52_values[start : end + 1]
        if window.shape != (LOOKBACK_HOURS, len(ENG52_FEATURE_NAMES)):
            incomplete_window += 1
            continue
        if not np.isfinite(window).all():
            nonfinite_window += 1
            continue
        endpoint_sigma = sigma[end]
        if not np.isfinite(endpoint_sigma) or endpoint_sigma == 0.0:
            invalid_sigma += 1
            continue
        if endpoint_sigma < 0.0:
            raise ValueError("rolling_volatility_24h cannot be negative")
        eligible_features.append(window[-1].copy())
        normalized_targets.append(float(targets[target_position] / endpoint_sigma))
        eligible_decisions.append(decision_time[target_position])

    features = pd.DataFrame(eligible_features, columns=ENG52_FEATURE_NAMES)
    normalized = np.asarray(normalized_targets, dtype=np.float64)
    if len(features) == 0 or not np.isfinite(normalized).all():
        raise ValueError("ENG52 screening requires finite eligible TRAIN samples")
    return ScreeningSamples(
        endpoint_features=features,
        normalized_targets=normalized,
        decision_times=pd.DatetimeIndex(eligible_decisions),
        candidate_count=len(ordered_targets),
        excluded_incomplete_window=incomplete_window,
        excluded_nonfinite_eng52_window=nonfinite_window,
        excluded_invalid_sigma=invalid_sigma,
        endpoint_missing_ratios=endpoint_missing_ratios,
    )


def chronological_blocks(sample_count: int, *, block_count: int = BLOCK_COUNT) -> list[np.ndarray]:
    if block_count != BLOCK_COUNT:
        raise ValueError("ENG52-S requires exactly six chronological blocks")
    if sample_count < block_count:
        raise ValueError("ENG52-S requires six non-empty chronological blocks")
    return [block for block in np.array_split(np.arange(sample_count), block_count)]


def _finite_summary(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(finite) == 0:
        return {
            "median_signed_spearman": None,
            "median_absolute_spearman": None,
            "minimum_absolute_spearman": None,
            "maximum_absolute_spearman": None,
            "spearman_standard_deviation": None,
        }
    absolute = np.abs(finite)
    return {
        "median_signed_spearman": float(np.median(finite)),
        "median_absolute_spearman": float(np.median(absolute)),
        "minimum_absolute_spearman": float(np.min(absolute)),
        "maximum_absolute_spearman": float(np.max(absolute)),
        "spearman_standard_deviation": float(np.std(finite, ddof=0)),
    }


def classify_spearman_signal(
    full_train_spearman: float | None,
    block_spearman: list[float | None],
) -> tuple[str, dict[str, object]]:
    if len(block_spearman) != BLOCK_COUNT:
        raise ValueError("Frozen candidate rule requires exactly six block correlations")
    positive = sum(value is not None and value > 0.0 for value in block_spearman)
    negative = sum(value is not None and value < 0.0 for value in block_spearman)
    if positive > negative:
        dominant_sign = "POSITIVE"
        sign_count = positive
        full_sign_matches = full_train_spearman is not None and full_train_spearman > 0.0
    elif negative > positive:
        dominant_sign = "NEGATIVE"
        sign_count = negative
        full_sign_matches = full_train_spearman is not None and full_train_spearman < 0.0
    else:
        dominant_sign = "NONE"
        sign_count = positive
        full_sign_matches = False
    summary: dict[str, object] = {
        "positive_block_count": positive,
        "negative_block_count": negative,
        "undefined_or_zero_block_count": BLOCK_COUNT - positive - negative,
        "dominant_sign": dominant_sign,
        "sign_consistent_block_count": sign_count,
        **_finite_summary(block_spearman),
    }
    median_absolute = summary["median_absolute_spearman"]
    stable = (
        sign_count >= MIN_SIGN_CONSISTENT_BLOCKS
        and full_sign_matches
        and median_absolute is not None
        and float(median_absolute) >= MIN_ABSOLUTE_SPEARMAN
        and full_train_spearman is not None
        and abs(full_train_spearman) >= MIN_ABSOLUTE_SPEARMAN
    )
    return (STABLE_CANDIDATE if stable else WEAK_OR_UNSTABLE), summary


def screen_features(samples: ScreeningSamples) -> tuple[dict[str, dict[str, object]], list[np.ndarray]]:
    blocks = chronological_blocks(len(samples.normalized_targets))
    reports: dict[str, dict[str, object]] = {}
    for name in ENG52_FEATURE_NAMES:
        values = samples.endpoint_features[name].to_numpy(np.float64)
        block_reports: list[dict[str, object]] = []
        block_spearman: list[float | None] = []
        for number, positions in enumerate(blocks, 1):
            spearman = _correlation(values[positions], samples.normalized_targets[positions], rank=True)
            block_spearman.append(spearman)
            block_reports.append(
                {
                    "block": number,
                    "start_decision_time": samples.decision_times[positions[0]].isoformat(),
                    "end_decision_time": samples.decision_times[positions[-1]].isoformat(),
                    "valid_sample_count": int(len(positions)),
                    "pearson": _correlation(
                        values[positions], samples.normalized_targets[positions], rank=False
                    ),
                    "spearman": spearman,
                }
            )
        full_spearman = _correlation(values, samples.normalized_targets, rank=True)
        classification, stability = classify_spearman_signal(full_spearman, block_spearman)
        reports[name] = {
            "full_train": {
                "valid_sample_count": int(len(values)),
                "pearson": _correlation(values, samples.normalized_targets, rank=False),
                "spearman": full_spearman,
            },
            "chronological_blocks": block_reports,
            "spearman_stability": stability,
            "train_endpoint_missing_ratio": samples.endpoint_missing_ratios[name],
            "screening_classification": classification,
        }
    if set(reports) != set(ENG52_FEATURE_NAMES) or len(reports) != 52:
        raise ValueError("Every one of the 52 ENG52 features must be classified")
    return reports, blocks


def _representative_key(report: dict[str, object]) -> tuple[int, float, float, float]:
    full = report["full_train"]
    stability = report["spearman_stability"]
    assert isinstance(full, dict) and isinstance(stability, dict)
    full_spearman = full["spearman"]
    median_absolute = stability["median_absolute_spearman"]
    return (
        1 if report["screening_classification"] == STABLE_CANDIDATE else 0,
        float(median_absolute) if median_absolute is not None else -np.inf,
        abs(float(full_spearman)) if full_spearman is not None else -np.inf,
        -float(report["train_endpoint_missing_ratio"]),
    )


def preferred_representative(
    left: str,
    right: str,
    reports: dict[str, dict[str, object]],
) -> tuple[str, str]:
    left_key = _representative_key(reports[left])
    right_key = _representative_key(reports[right])
    if left_key > right_key:
        return left, right
    if right_key > left_key:
        return right, left
    preferred = min(left, right)
    return preferred, right if preferred == left else left


def apply_redundancy_rule(
    reports: dict[str, dict[str, object]],
    qc_pairs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    pair_reports: list[dict[str, object]] = []
    alternates: dict[str, str] = {}
    for pair in qc_pairs:
        left = str(pair["left"])
        right = str(pair["right"])
        if left not in reports or right not in reports:
            raise ValueError(f"QC redundancy pair contains unknown feature: {left}, {right}")
        if abs(float(pair["spearman"])) < 0.98:
            raise ValueError("ENG52-S redundancy input must contain only >=0.98 pairs")
        preferred, alternate = preferred_representative(left, right, reports)
        previous = alternates.get(alternate)
        if previous is not None and previous != preferred:
            raise ValueError(f"Conflicting redundancy representatives for {alternate}")
        alternates[alternate] = preferred
        pair_reports.append(
            {
                "left": left,
                "right": right,
                "qc_spearman": float(pair["spearman"]),
                "left_screening_statistics": {
                    "classification": reports[left]["screening_classification"],
                    "full_train_spearman": reports[left]["full_train"]["spearman"],  # type: ignore[index]
                    "median_absolute_block_spearman": reports[left]["spearman_stability"]["median_absolute_spearman"],  # type: ignore[index]
                    "train_endpoint_missing_ratio": reports[left]["train_endpoint_missing_ratio"],
                },
                "right_screening_statistics": {
                    "classification": reports[right]["screening_classification"],
                    "full_train_spearman": reports[right]["full_train"]["spearman"],  # type: ignore[index]
                    "median_absolute_block_spearman": reports[right]["spearman_stability"]["median_absolute_spearman"],  # type: ignore[index]
                    "train_endpoint_missing_ratio": reports[right]["train_endpoint_missing_ratio"],
                },
                "preferred_representative": preferred,
                "non_preferred_member": alternate,
                "non_preferred_classification": REDUNDANT_ALTERNATE,
            }
        )
    return pair_reports, alternates


def sparse_feature_report(
    samples: ScreeningSamples,
    blocks: list[np.ndarray],
) -> dict[str, object]:
    report: dict[str, object] = {}
    for name in SPARSE_FEATURE_NAMES:
        values = samples.endpoint_features[name].to_numpy(np.float64)
        counts = [int(np.count_nonzero(values[positions] != 0.0)) for positions in blocks]
        report[name] = {
            "train_nonzero_event_ratio": float(np.mean(values != 0.0)),
            "events_by_block": counts,
            "temporal_sparsity_flag": "TEMPORALLY_SPARSE" if 0 in counts else None,
        }
    return report


def _selection_lists(
    reports: dict[str, dict[str, object]],
    alternates: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    stable = sorted(
        name
        for name, report in reports.items()
        if report["screening_classification"] == STABLE_CANDIDATE and name not in alternates
    )
    weak = sorted(
        name
        for name, report in reports.items()
        if report["screening_classification"] == WEAK_OR_UNSTABLE and name not in alternates
    )
    redundant = sorted(alternates)
    if set(stable) | set(weak) | set(redundant) != set(ENG52_FEATURE_NAMES):
        raise ValueError("Screening output categories must cover all 52 ENG52 features")
    return stable, redundant, weak


def _write_outputs(
    *,
    screening_path: Path,
    config_path: Path,
    report: dict[str, object],
    config: dict[str, object],
) -> None:
    screening_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    screening_temporary = screening_path.with_suffix(f"{screening_path.suffix}.tmp")
    config_temporary = config_path.with_suffix(f"{config_path.suffix}.tmp")
    screening_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_temporary.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    os.replace(screening_temporary, screening_path)
    os.replace(config_temporary, config_path)


def run_eng52_signal_screening(*, project_root: Path) -> Eng52ScreeningResult:
    root = project_root.resolve()
    split_metadata = load_yaml(root / FROZEN_SPLIT_RELATIVE_PATH)
    train_targets = _load_train_targets(
        root / TARGET_RELATIVE_PATH, split_metadata=split_metadata
    )
    maximum_anchor = pd.to_datetime(train_targets["bar_open_time"], utc=True).max()
    eng52 = pq.read_table(
        root / ENG52_OUTPUT_RELATIVE_PATH,
        columns=["open_time", *ENG52_FEATURE_NAMES],
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    ).to_pandas()
    canonical = pq.read_table(
        root / CANONICAL_1H_RELATIVE_PATH,
        filters=[("open_time", "<=", maximum_anchor.to_pydatetime())],
    ).to_pandas()
    f0 = compute_f0_features(canonical)
    samples = build_screening_samples(eng52, f0, train_targets)
    feature_reports, blocks = screen_features(samples)

    qc = json.loads((root / QC_SUMMARY_RELATIVE_PATH).read_text(encoding="utf-8"))
    qc_pairs = qc["redundancy_train_only"]["absolute_spearman_gte_0_98_pairs"]
    if not isinstance(qc_pairs, list):
        raise ValueError("ENG52-QC redundancy pairs must be a list")
    qc_sparse = set(qc["numerical_flags"]["sparse_or_binary_feature_names"])
    if not set(SPARSE_FEATURE_NAMES).issubset(qc_sparse):
        raise ValueError("ENG52-QC sparse flags do not contain the frozen sparse features")
    redundancy, alternates = apply_redundancy_rule(feature_reports, qc_pairs)
    stable, redundant, weak = _selection_lists(feature_reports, alternates)

    block_boundaries = [
        {
            "block": number,
            "start_decision_time": samples.decision_times[positions[0]].isoformat(),
            "end_decision_time": samples.decision_times[positions[-1]].isoformat(),
            "sample_count": int(len(positions)),
        }
        for number, positions in enumerate(blocks, 1)
    ]
    rule = {
        "required_same_sign_blocks": MIN_SIGN_CONSISTENT_BLOCKS,
        "total_chronological_blocks": BLOCK_COUNT,
        "full_train_sign_must_match_dominant_block_sign": True,
        "minimum_median_absolute_block_spearman": MIN_ABSOLUTE_SPEARMAN,
        "minimum_absolute_full_train_spearman": MIN_ABSOLUTE_SPEARMAN,
        "otherwise": WEAK_OR_UNSTABLE,
    }
    report: dict[str, object] = {
        "screening_id": "ENG52-S",
        "scope": {
            "target_rows": "frozen TRAIN only",
            "validation_targets": "NOT READ OR USED",
            "test_targets": "NOT READ OR USED",
            "model_training": False,
        },
        "source_eng52_artifact": ENG52_OUTPUT_RELATIVE_PATH.as_posix(),
        "source_qc_artifact": QC_SUMMARY_RELATIVE_PATH.as_posix(),
        "target_definition": {
            "sigma_t": VOLATILITY_FEATURE_NAME,
            "z_t": "future_log_return_1h / sigma_t",
            "epsilon_added": False,
            "sigma_alignment": "sequence endpoint available at decision time",
        },
        "sample_coverage": {
            "candidate_frozen_train_rows": samples.candidate_count,
            "eligible_train_rows": len(samples.normalized_targets),
            "excluded_incomplete_or_nonconsecutive_24h_window": samples.excluded_incomplete_window,
            "excluded_nonfinite_eng52_window": samples.excluded_nonfinite_eng52_window,
            "excluded_invalid_sigma": samples.excluded_invalid_sigma,
        },
        "chronological_train_blocks": block_boundaries,
        "screening_rule": rule,
        "features": feature_reports,
        "redundancy": redundancy,
        "sparse_features": sparse_feature_report(samples, blocks),
        "selected_feature_groups": {
            "stable_candidates": stable,
            "redundant_alternates": redundant,
            "weak_or_unstable": weak,
        },
    }
    config: dict[str, object] = {
        "feature_set": {
            "id": "ENG52_STABLE_CANDIDATES",
            "status": "train_only_temporal_screening",
            "source_eng52_artifact": ENG52_OUTPUT_RELATIVE_PATH.as_posix(),
            "source_qc_artifact": QC_SUMMARY_RELATIVE_PATH.as_posix(),
            "train_only_scope": (
                "Frozen TRAIN targets only; validation and TEST target rows are neither read nor used."
            ),
            "target": {
                "definition": "future_log_return_1h / rolling_volatility_24h",
                "epsilon_added": False,
            },
            "screening_rule": rule,
            "stable_candidate_features": stable,
            "redundant_alternates": [
                {
                    "feature_name": name,
                    "preferred_representative": alternates[name],
                }
                for name in redundant
            ],
            "weak_or_unstable_features": weak,
        }
    }
    screening_path = root / SCREENING_OUTPUT_RELATIVE_PATH
    config_path = root / STABLE_CONFIG_RELATIVE_PATH
    _write_outputs(
        screening_path=screening_path,
        config_path=config_path,
        report=report,
        config=config,
    )
    return Eng52ScreeningResult(screening_path, config_path, report)
