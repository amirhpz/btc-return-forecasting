from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from btc_forecasting.baselines.ridge import _retained_target_splits
from btc_forecasting.common.config import load_yaml
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES, compute_eng52_features
from btc_forecasting.features.eng52_build import ENG52_OUTPUT_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_RELATIVE_PATH

QC_SUMMARY_RELATIVE_PATH = Path("outputs/data/eng52/qc_summary.json")
EXPECTED_E03_VN_VALIDATION_IDENTITY = (
    "3907f80c1b59c98d13d1a733953e7a679b528747b953d3e9db58c63cc10ba13c"
)
LOOKBACK_HOURS = 24
ONE_HOUR = timedelta(hours=1)
TARGET_COLUMNS = ("bar_open_time", "decision_time", "target_time", "future_log_return_1h")
STATEFUL_FEATURES = (
    "mom_tl_break_bull_30",
    "rsi_div_persistence",
    "rsi_hidden_div_flag",
)


@dataclass(frozen=True)
class Eng52QcResult:
    artifact_path: Path
    summary: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _sample_identity(decision_times: pd.DatetimeIndex) -> str:
    values = decision_times.asi8.astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _numerical_statistics(features: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    statistics: dict[str, object] = {}
    constants: list[str] = []
    near_constants: list[str] = []
    sparse_or_binary: list[str] = []
    extreme_tails: list[str] = []
    total_inf = 0
    for name in ENG52_FEATURE_NAMES:
        values = features[name].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        inf_count = int(np.count_nonzero(np.isinf(values)))
        total_inf += inf_count
        if len(finite) == 0:
            raise ValueError(f"ENG52 feature has no finite values: {name}")
        unique, counts = np.unique(finite, return_counts=True)
        quantiles = np.quantile(finite, [0.001, 0.01, 0.5, 0.99, 0.999])
        zero_ratio = float(np.count_nonzero(finite == 0.0) / len(finite))
        dominant_ratio = float(counts.max() / len(finite))
        constant = len(unique) == 1
        near_constant = not constant and dominant_ratio >= 0.99
        sparse = len(unique) <= 2 or zero_ratio >= 0.95
        outer_reference = max(abs(float(quantiles[0])), abs(float(quantiles[-1])), 1e-15)
        extreme_tail = max(abs(float(finite.min())), abs(float(finite.max()))) > 100.0 * outer_reference
        if constant:
            constants.append(name)
        if near_constant:
            near_constants.append(name)
        if sparse:
            sparse_or_binary.append(name)
        if extreme_tail:
            extreme_tails.append(name)
        statistics[name] = {
            "finite_count": int(len(finite)),
            "missing_count": int(np.count_nonzero(np.isnan(values))),
            "inf_count": inf_count,
            "unique_finite_value_count": int(len(unique)),
            "zero_ratio_among_finite": zero_ratio,
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "min": float(finite.min()),
            "q001": float(quantiles[0]),
            "q01": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q99": float(quantiles[3]),
            "q999": float(quantiles[4]),
            "max": float(finite.max()),
            "flags": {
                "constant": constant,
                "near_constant_dominant_value_ratio_gte_0_99": near_constant,
                "sparse_or_binary_zero_ratio_gte_0_95_or_unique_lte_2": sparse,
                "extreme_tail_outer_value_gt_100x_q001_q999_scale": extreme_tail,
            },
        }
    flags: dict[str, object] = {
        "constant_feature_names": constants,
        "near_constant_feature_names": near_constants,
        "sparse_or_binary_feature_names": sparse_or_binary,
        "extreme_tail_feature_names": extreme_tails,
        "total_inf_count": total_inf,
    }
    return statistics, flags


def _redundancy(train_features: pd.DataFrame) -> dict[str, object]:
    exact_pairs: list[dict[str, object]] = []
    high_spearman_pairs: list[dict[str, object]] = []
    correlation = train_features.corr(method="spearman", min_periods=2)
    for left_index, left in enumerate(ENG52_FEATURE_NAMES):
        left_values = train_features[left].to_numpy(float)
        for right in ENG52_FEATURE_NAMES[left_index + 1 :]:
            right_values = train_features[right].to_numpy(float)
            exact = bool(np.array_equal(left_values, right_values, equal_nan=True))
            if exact:
                exact_pairs.append({"left": left, "right": right})
            value = float(correlation.loc[left, right])
            if np.isfinite(value) and abs(value) >= 0.98:
                pairwise = int(np.count_nonzero(np.isfinite(left_values) & np.isfinite(right_values)))
                high_spearman_pairs.append(
                    {"left": left, "right": right, "spearman": value, "pairwise_finite_count": pairwise}
                )
    return {
        "exact_duplicate_pairs": exact_pairs,
        "absolute_spearman_gte_0_98_pairs": high_spearman_pairs,
    }


def _coverage(
    feature_rows: pd.DataFrame,
    feature_names: tuple[str, ...],
    targets: pd.DataFrame,
) -> dict[str, object]:
    feature_time = pd.DatetimeIndex(pd.to_datetime(feature_rows["open_time"], utc=True))
    values = feature_rows.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    ordered_targets = targets.sort_values("decision_time", kind="stable").reset_index(drop=True)
    bar_time = pd.DatetimeIndex(pd.to_datetime(ordered_targets["bar_open_time"], utc=True))
    decision_time = pd.DatetimeIndex(pd.to_datetime(ordered_targets["decision_time"], utc=True))
    positions = feature_time.get_indexer(bar_time)
    usable_decisions: list[pd.Timestamp] = []
    incomplete_window = 0
    nonfinite_features = 0
    for target_position, end in enumerate(positions):
        start = end - LOOKBACK_HOURS + 1
        if end < 0 or start < 0 or feature_time[end] - feature_time[start] != 23 * ONE_HOUR:
            incomplete_window += 1
            continue
        window = values[start : end + 1]
        if window.shape != (LOOKBACK_HOURS, len(feature_names)):
            incomplete_window += 1
            continue
        if not np.isfinite(window).all():
            nonfinite_features += 1
            continue
        usable_decisions.append(decision_time[target_position])
    identity = _sample_identity(pd.DatetimeIndex(usable_decisions))
    return {
        "candidate_target_rows": len(ordered_targets),
        "usable_samples": len(usable_decisions),
        "excluded_missing_or_nonfinite_features": nonfinite_features,
        "excluded_incomplete_or_nonconsecutive_24h_window": incomplete_window,
        "sample_identity_sha256": identity,
        "matches_e03_vn_mse_validation_identity": identity == EXPECTED_E03_VN_VALIDATION_IDENTITY,
    }


def _stateful_gap_checks(canonical: pd.DataFrame, artifact: pd.DataFrame) -> dict[str, object]:
    times = pd.DatetimeIndex(pd.to_datetime(canonical["open_time"], utc=True))
    segment_starts = np.flatnonzero(times.to_series(index=np.arange(len(times))).diff().ne(ONE_HOUR).to_numpy())
    boundaries = [*segment_starts.tolist(), len(canonical)]
    matches = {name: True for name in STATEFUL_FEATURES}
    first_values_zero = {name: True for name in STATEFUL_FEATURES}
    for boundary_index, start in enumerate(boundaries[:-1]):
        stop = boundaries[boundary_index + 1]
        standalone = compute_eng52_features(canonical.iloc[start:stop].reset_index(drop=True))
        for name in STATEFUL_FEATURES:
            expected = standalone[name].to_numpy(float)
            actual = artifact[name].iloc[start:stop].to_numpy(float)
            matches[name] &= bool(np.array_equal(actual, expected, equal_nan=True))
            first_values_zero[name] &= bool(actual[0] == 0.0)
    semantics = {
        "mom_tl_break_bull_30": "Zero means no confirmed bullish trendline-break event; it is defined from segment start.",
        "rsi_div_persistence": "Zero means no active hidden-divergence run; it is defined from segment start.",
        "rsi_hidden_div_flag": "Zero is the documented neutral state; it is defined before two confirmed pivots exist.",
    }
    return {
        "real_gap_count": max(len(segment_starts) - 1, 0),
        "features": {
            name: {
                "artifact_matches_standalone_segment_recomputation": matches[name],
                "first_row_of_every_segment_is_zero": first_values_zero[name],
                "missing_count": int(artifact[name].isna().sum()),
                "zero_semantics": semantics[name],
                "conclusion": "State resets at every gap; zero is an explicit defined neutral/no-event value, not a missing-history substitute.",
            }
            for name in STATEFUL_FEATURES
        },
    }


def run_eng52_qc(*, project_root: Path) -> Eng52QcResult:
    root = project_root.resolve()
    artifact = pd.read_parquet(root / ENG52_OUTPUT_RELATIVE_PATH)
    if tuple(artifact.columns) != ("open_time", *ENG52_FEATURE_NAMES):
        raise ValueError("ENG52 artifact schema does not match the frozen feature order")
    canonical = pd.read_parquet(root / CANONICAL_1H_RELATIVE_PATH)
    if not pd.to_datetime(artifact["open_time"], utc=True).equals(
        pd.to_datetime(canonical["open_time"], utc=True)
    ):
        raise ValueError("ENG52 artifact timestamps do not match canonical 1-hour data")

    split_metadata = load_yaml(root / FROZEN_SPLIT_RELATIVE_PATH)
    test_start = _parse_utc(
        split_metadata["split"]["boundaries"]["test"]["decision_time_start_inclusive"]
    )
    target_rows = pq.read_table(
        root / TARGET_RELATIVE_PATH,
        columns=list(TARGET_COLUMNS),
        filters=[("decision_time", "<", test_start)],
    ).to_pandas()
    train_targets, validation_targets = _retained_target_splits(
        target_rows, split_metadata=split_metadata
    )
    train_anchor = pd.DatetimeIndex(pd.to_datetime(train_targets["bar_open_time"], utc=True))
    artifact_by_time = artifact.set_index(pd.to_datetime(artifact["open_time"], utc=True))
    train_eng52 = artifact_by_time.reindex(train_anchor).loc[:, ENG52_FEATURE_NAMES]

    numerical, flags = _numerical_statistics(artifact)
    redundancy = _redundancy(train_eng52)
    f0 = compute_f0_features(canonical)
    combined = f0.merge(artifact, on="open_time", how="inner", validate="one_to_one")
    coverage = {
        "F0": {
            "train": _coverage(f0, F0_FEATURE_NAMES, train_targets),
            "validation": _coverage(f0, F0_FEATURE_NAMES, validation_targets),
            "expected_reference_counts": {"train": 53343, "validation": 11745},
        },
        "ENG52": {
            "train": _coverage(artifact, ENG52_FEATURE_NAMES, train_targets),
            "validation": _coverage(artifact, ENG52_FEATURE_NAMES, validation_targets),
        },
        "F0_plus_ENG52": {
            "train": _coverage(combined, (*F0_FEATURE_NAMES, *ENG52_FEATURE_NAMES), train_targets),
            "validation": _coverage(combined, (*F0_FEATURE_NAMES, *ENG52_FEATURE_NAMES), validation_targets),
        },
    }
    if (
        coverage["F0"]["train"]["usable_samples"] != 53343  # type: ignore[index]
        or coverage["F0"]["validation"]["usable_samples"] != 11745  # type: ignore[index]
    ):
        raise ValueError("F0 coverage no longer matches the frozen E03 reference")

    summary: dict[str, object] = {
        "build_warning": {
            "source": "return_skew_30 pandas rolling skew on a contiguous segment containing fewer than 30 finite one-hour returns",
            "condition": "A short post-gap segment has only missing rolling-skew windows because its exact 30-return history never becomes available.",
            "classification": "expected NaN from frozen gap/warm-up semantics",
            "action": "Return the same all-missing series without calling pandas rolling skew when a segment cannot contain one mature window.",
        },
        "stateful_gap_checks": _stateful_gap_checks(canonical, artifact),
        "numerical_qc": numerical,
        "numerical_flags": flags,
        "redundancy_train_only": redundancy,
        "coverage": coverage,
        "test_set_predictive_evaluation": "NOT ACCESSED",
    }
    output_path = root / QC_SUMMARY_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return Eng52QcResult(output_path, summary)
