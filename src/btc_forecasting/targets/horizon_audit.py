from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from btc_forecasting.common.config import load_yaml
from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES, compute_f0_features
from btc_forecasting.splits.frozen import FROZEN_SPLIT_RELATIVE_PATH
from btc_forecasting.targets.one_hour import TARGET_COLUMN, TARGET_RELATIVE_PATH
from btc_forecasting.training.lstm_generalization import _correlation

AUDIT_OUTPUT_RELATIVE_PATH = Path("outputs/data/horizon_audit/summary.json")
FROZEN_HORIZONS_HOURS = (1, 3, 6, 12)
LOOKBACK_HOURS = 24
ONE_HOUR = timedelta(hours=1)
VOLATILITY_FEATURE_NAME = "rolling_volatility_24h"
EXISTING_TARGET_COLUMNS = (
    "bar_open_time",
    "decision_time",
    "target_time",
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class HorizonConstruction:
    horizon_hours: int
    valid: pd.DataFrame
    counts: dict[str, int]


@dataclass(frozen=True)
class HorizonAuditResult:
    artifact_path: Path
    summary: dict[str, object]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Frozen split timestamp must be UTC: {value!r}")
    return parsed


def _iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _sample_identity(decision_times: pd.DatetimeIndex) -> str:
    values = decision_times.asi8.astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def construct_horizon_targets(
    canonical: pd.DataFrame,
    anchor_times: pd.DatetimeIndex,
    *,
    horizon_hours: int,
    target_scope_end_exclusive: datetime,
) -> HorizonConstruction:
    if horizon_hours not in FROZEN_HORIZONS_HOURS:
        raise ValueError(f"Unsupported frozen horizon: {horizon_hours}h")
    if not {"open_time", "close"}.issubset(canonical.columns):
        raise ValueError("Canonical source must contain open_time and close")
    times = pd.DatetimeIndex(pd.to_datetime(canonical["open_time"], utc=True))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Canonical timestamps must be strictly ordered and unique")
    anchors = pd.DatetimeIndex(pd.to_datetime(anchor_times, utc=True))
    if anchors.has_duplicates or not anchors.is_monotonic_increasing:
        raise ValueError("Target anchors must be strictly ordered and unique")
    closes = canonical["close"].to_numpy(np.float64)
    position_by_time = {timestamp.value: index for index, timestamp in enumerate(times)}
    scope_end = pd.Timestamp(target_scope_end_exclusive)
    if scope_end.tzinfo is None:
        scope_end = scope_end.tz_localize("UTC")
    else:
        scope_end = scope_end.tz_convert("UTC")

    rows: list[dict[str, object]] = []
    missing_endpoint = 0
    nonconsecutive_path = 0
    boundary_crossing = 0
    nonfinite = 0
    for anchor in anchors:
        expected_endpoint = anchor + horizon_hours * ONE_HOUR
        target_time = expected_endpoint + ONE_HOUR
        if target_time >= scope_end:
            boundary_crossing += 1
            continue
        endpoint_position = position_by_time.get(expected_endpoint.value)
        if endpoint_position is None:
            missing_endpoint += 1
            continue
        required_times = [anchor + offset * ONE_HOUR for offset in range(horizon_hours + 1)]
        path_complete = all(timestamp.value in position_by_time for timestamp in required_times)
        if not path_complete:
            nonconsecutive_path += 1
            continue
        anchor_position = position_by_time.get(anchor.value)
        if anchor_position is None:
            raise ValueError("Target anchor is absent from canonical source")
        current_price = float(closes[anchor_position])
        endpoint_price = float(closes[endpoint_position])
        if (
            not np.isfinite(current_price)
            or not np.isfinite(endpoint_price)
            or current_price <= 0.0
            or endpoint_price <= 0.0
        ):
            nonfinite += 1
            continue
        target = math.log(endpoint_price / current_price)
        if not np.isfinite(target):
            nonfinite += 1
            continue
        rows.append(
            {
                "feature_bar_time": anchor,
                "decision_time": anchor + ONE_HOUR,
                "price_time": anchor,
                "current_price": current_price,
                "expected_endpoint_time": expected_endpoint,
                "actual_endpoint_time": times[endpoint_position],
                "endpoint_price": endpoint_price,
                "target_time": target_time,
                "future_log_return": target,
                "complete_consecutive_path": path_complete,
            }
        )
    valid = pd.DataFrame(rows)
    if not valid.empty:
        valid = valid.sort_values("decision_time", kind="stable").reset_index(drop=True)
    counts = {
        "target_candidates": len(anchors),
        "valid_targets": len(valid),
        "missing_endpoint_exclusions": missing_endpoint,
        "nonconsecutive_future_path_exclusions": nonconsecutive_path,
        "train_boundary_crossing_exclusions": boundary_crossing,
        "nonfinite_exclusions": nonfinite,
    }
    if sum(value for key, value in counts.items() if key != "target_candidates") != len(anchors):
        raise RuntimeError("Horizon exclusion accounting does not reconcile")
    return HorizonConstruction(horizon_hours, valid, counts)


def f0_eligible_anchor_data(
    f0: pd.DataFrame,
    anchor_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {"open_time", *F0_FEATURE_NAMES}
    missing = sorted(required - set(f0.columns))
    if missing:
        raise ValueError(f"Missing F0 columns: {missing}")
    times = pd.DatetimeIndex(pd.to_datetime(f0["open_time"], utc=True))
    values = f0.loc[:, list(F0_FEATURE_NAMES)].to_numpy(np.float64)
    positions = times.get_indexer(pd.DatetimeIndex(anchor_times))
    rows: list[dict[str, object]] = []
    for end in positions:
        start = end - LOOKBACK_HOURS + 1
        if end < 0 or start < 0 or times[end] - times[start] != 23 * ONE_HOUR:
            continue
        window = values[start : end + 1]
        if window.shape != (LOOKBACK_HOURS, len(F0_FEATURE_NAMES)):
            continue
        if not np.isfinite(window).all():
            continue
        sigma = float(window[-1, F0_FEATURE_NAMES.index(VOLATILITY_FEATURE_NAME)])
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        rows.append(
            {
                "feature_bar_time": times[end],
                "decision_time": times[end] + ONE_HOUR,
                VOLATILITY_FEATURE_NAME: sigma,
            }
        )
    return pd.DataFrame(rows)


def common_anchor_times(
    f0_eligible_decisions: pd.DatetimeIndex,
    valid_decisions_by_horizon: dict[int, pd.DatetimeIndex],
) -> pd.DatetimeIndex:
    if tuple(sorted(valid_decisions_by_horizon)) != FROZEN_HORIZONS_HOURS:
        raise ValueError("Common anchors require all four frozen horizons")
    common = set(pd.DatetimeIndex(f0_eligible_decisions).asi8.tolist())
    for horizon in FROZEN_HORIZONS_HOURS:
        common &= set(pd.DatetimeIndex(valid_decisions_by_horizon[horizon]).asi8.tolist())
    return pd.to_datetime(sorted(common), utc=True)


def raw_distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Raw target distribution requires finite non-empty values")
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "mean_absolute_return": float(np.mean(np.abs(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "positive_fraction": float(np.mean(array > 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "same_row_zero_predictor_mae": float(np.mean(np.abs(array))),
    }


def normalized_diagnostic(
    targets: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float | int]:
    raw = np.asarray(targets, dtype=np.float64)
    volatility = np.asarray(sigma, dtype=np.float64)
    if raw.shape != volatility.shape or raw.size == 0:
        raise ValueError("q(H) requires equal non-empty target and sigma arrays")
    if not np.isfinite(raw).all() or not np.isfinite(volatility).all():
        raise ValueError("q(H) requires finite targets and sigma")
    if np.any(volatility <= 0.0):
        raise ValueError("q(H) requires sigma strictly greater than zero")
    values = raw / volatility
    quantiles = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "median_absolute_value": float(np.median(np.abs(values))),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "epsilon_added": False,
    }


def _select_alignment_positions(count: int) -> list[int]:
    if count < 9:
        raise ValueError("Alignment verification requires at least nine eligible anchors")
    middle = count // 2
    return [0, 1, 2, middle - 1, middle, middle + 1, count - 3, count - 2, count - 1]


def alignment_checks(construction: HorizonConstruction) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    horizon_delta = construction.horizon_hours * ONE_HOUR
    for position in _select_alignment_positions(len(construction.valid)):
        row = construction.valid.iloc[position]
        price_time = pd.Timestamp(row["price_time"])
        endpoint_time = pd.Timestamp(row["actual_endpoint_time"])
        expected = pd.Timestamp(row["expected_endpoint_time"])
        recomputed = math.log(float(row["endpoint_price"]) / float(row["current_price"]))
        aligned = endpoint_time - price_time == horizon_delta and endpoint_time == expected
        equal = math.isclose(
            recomputed,
            float(row["future_log_return"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        if not aligned or not equal or not bool(row["complete_consecutive_path"]):
            raise ValueError("Horizon alignment verification failed")
        checks.append(
            {
                "feature_bar_timestamp": _iso(row["feature_bar_time"]),
                "decision_time": _iso(row["decision_time"]),
                "price_t_timestamp": _iso(price_time),
                "price_t": float(row["current_price"]),
                "expected_endpoint_timestamp": _iso(expected),
                "actual_endpoint_timestamp": _iso(endpoint_time),
                "price_t_plus_h": float(row["endpoint_price"]),
                "calculated_log_return": float(row["future_log_return"]),
                "independently_recomputed_log_return": recomputed,
                "all_required_hourly_timestamps_exist": True,
                "endpoint_minus_price_t_hours": construction.horizon_hours,
            }
        )
    return checks


def _read_existing_train_targets(
    path: Path,
    *,
    train_start: datetime,
    validation_start: datetime,
    target_end: datetime,
) -> pd.DataFrame:
    return pq.read_table(
        path,
        columns=list(EXISTING_TARGET_COLUMNS),
        filters=[
            ("decision_time", ">=", train_start),
            ("decision_time", "<", validation_start),
            ("target_time", "<", target_end),
        ],
    ).to_pandas()


def one_hour_regression_check(
    construction: HorizonConstruction,
    existing: pd.DataFrame,
) -> dict[str, object]:
    if construction.horizon_hours != 1:
        raise ValueError("Regression check requires H=1h construction")
    new = construction.valid.set_index("decision_time").sort_index()
    old = existing.copy()
    old.index = pd.DatetimeIndex(pd.to_datetime(old["decision_time"], utc=True))
    old = old.sort_index()
    symmetric_difference = new.index.symmetric_difference(old.index)
    common = new.index.intersection(old.index)
    timestamp_mismatches = len(symmetric_difference)
    if len(common):
        timestamp_mismatches += int(
            np.count_nonzero(
                pd.to_datetime(new.loc[common, "feature_bar_time"], utc=True).array
                != pd.to_datetime(old.loc[common, "bar_open_time"], utc=True).array
            )
        )
        timestamp_mismatches += int(
            np.count_nonzero(
                pd.to_datetime(new.loc[common, "target_time"], utc=True).array
                != pd.to_datetime(old.loc[common, "target_time"], utc=True).array
            )
        )
    differences = np.abs(
        new.loc[common, "future_log_return"].to_numpy(np.float64)
        - old.loc[common, TARGET_COLUMN].to_numpy(np.float64)
    )
    maximum_difference = float(np.max(differences)) if len(differences) else math.inf
    equal = bool(
        len(common) == len(new) == len(old)
        and np.allclose(
            new.loc[common, "future_log_return"].to_numpy(np.float64),
            old.loc[common, TARGET_COLUMN].to_numpy(np.float64),
            rtol=1e-12,
            atol=1e-15,
        )
    )
    if timestamp_mismatches != 0 or not equal:
        raise ValueError("H=1h target logic does not reproduce frozen TRAIN semantics")
    return {
        "common_row_count": len(common),
        "maximum_absolute_target_difference": maximum_difference,
        "timestamp_mismatch_count": timestamp_mismatches,
        "numerically_equal_within_tolerance": equal,
        "relative_tolerance": 1e-12,
        "absolute_tolerance": 1e-15,
    }


def _values_on_decisions(
    construction: HorizonConstruction,
    decisions: pd.DatetimeIndex,
) -> np.ndarray:
    indexed = construction.valid.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed["decision_time"], utc=True))
    values = indexed.reindex(decisions)["future_log_return"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Requested target scope contains a missing/nonfinite value")
    return values


def _sigma_on_decisions(
    f0_eligible: pd.DataFrame,
    decisions: pd.DatetimeIndex,
) -> np.ndarray:
    indexed = f0_eligible.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed["decision_time"], utc=True))
    sigma = indexed.reindex(decisions)[VOLATILITY_FEATURE_NAME].to_numpy(np.float64)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("Claimed F0-eligible anchor has invalid sigma")
    return sigma


def _matrix(frame: pd.DataFrame, *, method: str) -> dict[str, dict[str, float]]:
    correlation = frame.corr(method=method)
    return {
        str(row): {str(column): float(correlation.loc[row, column]) for column in frame}
        for row in frame
    }


def _lag_one_autocorrelation(
    decisions: pd.DatetimeIndex,
    values: np.ndarray,
) -> dict[str, float | int | None]:
    consecutive = (decisions[1:] - decisions[:-1]) == ONE_HOUR
    left = values[:-1][consecutive]
    right = values[1:][consecutive]
    return {
        "consecutive_real_hour_pair_count": int(len(left)),
        "pearson_lag_1": _correlation(left, right, rank=False),
    }


def run_horizon_target_audit(*, project_root: Path) -> HorizonAuditResult:
    root = project_root.resolve()
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
    f0_eligible = f0_eligible_anchor_data(f0, anchor_times)
    f0_decisions = pd.DatetimeIndex(
        pd.to_datetime(f0_eligible["decision_time"], utc=True)
    )
    valid_decisions = {
        horizon: pd.DatetimeIndex(
            pd.to_datetime(construction.valid["decision_time"], utc=True)
        )
        for horizon, construction in constructions.items()
    }
    common = common_anchor_times(f0_decisions, valid_decisions)
    if len(common) == 0:
        raise ValueError("No common paired horizon anchors are available")

    horizon_reports: dict[str, object] = {}
    common_targets: dict[str, np.ndarray] = {}
    common_raw_std: dict[int, float] = {}
    for horizon, construction in constructions.items():
        native = f0_decisions.intersection(valid_decisions[horizon]).sort_values()
        native_targets = _values_on_decisions(construction, native)
        common_values = _values_on_decisions(construction, common)
        common_targets[f"{horizon}h"] = common_values
        native_sigma = _sigma_on_decisions(f0_eligible, native)
        common_sigma = _sigma_on_decisions(f0_eligible, common)
        native_raw = raw_distribution(native_targets)
        common_raw = raw_distribution(common_values)
        common_raw_std[horizon] = float(common_raw["standard_deviation"])
        horizon_reports[f"{horizon}h"] = {
            "coverage": {
                **construction.counts,
                "f0_model_eligible_targets": len(native),
                "loss_from_target_candidates_to_valid_target": (
                    construction.counts["target_candidates"]
                    - construction.counts["valid_targets"]
                ),
                "loss_from_valid_target_to_f0_eligible": (
                    construction.counts["valid_targets"] - len(native)
                ),
                "loss_from_native_f0_eligible_to_common": len(native) - len(common),
                "common_paired_targets": len(common),
            },
            "native_f0_eligible": {
                "raw_return": native_raw,
                "q_target_over_sigma_24h": normalized_diagnostic(
                    native_targets,
                    native_sigma,
                ),
            },
            "common_paired": {
                "raw_return": common_raw,
                "q_target_over_sigma_24h": normalized_diagnostic(
                    common_values,
                    common_sigma,
                ),
            },
            "alignment_checks": alignment_checks(construction),
        }

    right_edge_counts = [
        constructions[horizon].counts["train_boundary_crossing_exclusions"]
        for horizon in FROZEN_HORIZONS_HOURS
    ]
    expected_right_edge_counts = [
        int(
            np.count_nonzero(
                anchor_times + (horizon + 1) * ONE_HOUR >= pd.Timestamp(target_end)
            )
        )
        for horizon in FROZEN_HORIZONS_HOURS
    ]
    if (
        right_edge_counts != expected_right_edge_counts
        or right_edge_counts != sorted(right_edge_counts)
    ):
        raise ValueError("Longer horizons did not produce monotonic right-edge purging")
    one_hour_std = common_raw_std[1]
    variance_ratios = {
        f"{horizon}h": {
            "raw_std_over_1h_raw_std": common_raw_std[horizon] / one_hour_std,
            "raw_std_over_sqrt_h_times_1h_raw_std": (
                common_raw_std[horizon] / (math.sqrt(horizon) * one_hour_std)
            ),
        }
        for horizon in (3, 6, 12)
    }
    common_frame = pd.DataFrame(common_targets, index=common)
    existing = _read_existing_train_targets(
        root / TARGET_RELATIVE_PATH,
        train_start=train_start,
        validation_start=validation_start,
        target_end=target_end,
    )
    regression = one_hour_regression_check(constructions[1], existing)
    summary: dict[str, object] = {
        "audit_id": "E04-H-A",
        "scope": {
            "source": "frozen TRAIN only",
            "original_validation": "NOT READ OR USED",
            "test": "NOT READ OR USED",
            "latest_canonical_open_time_read": _iso(canonical_times[-1]),
        },
        "frozen_horizons_hours": list(FROZEN_HORIZONS_HOURS),
        "target_semantics": {
            "formula": "log(close_of_bar_opened_at_t_plus_H / close_of_bar_opened_at_t)",
            "feature_bar_time": "t",
            "decision_time": "t + 1h",
            "endpoint_bar_open_time": "t + H",
            "target_time": "t + H + 1h",
            "continuity": "all hourly opens t through t+H must exist exactly",
            "boundary": "target_time must be strictly before frozen TRAIN target end",
            "interpolation_or_fill": False,
        },
        "target_anchor_candidates": len(anchor_times),
        "f0_eligible_anchor_count_before_horizon_target_intersection": len(f0_decisions),
        "common_paired_anchors": {
            "sample_count": len(common),
            "first_decision_time": _iso(common[0]),
            "last_decision_time": _iso(common[-1]),
            "sample_identity_sha256": _sample_identity(common),
            "requirements": [
                "F0 24h consecutive model sequence",
                "finite positive sigma_24h",
                "valid 1h target",
                "valid 3h target",
                "valid 6h target",
                "valid 12h target",
            ],
        },
        "horizons": horizon_reports,
        "empirical_common_anchor_horizon_scale_ratios": variance_ratios,
        "horizon_relationships_on_common_anchors": {
            "pearson_matrix": _matrix(common_frame, method="pearson"),
            "spearman_matrix": _matrix(common_frame, method="spearman"),
            "lag_1_autocorrelation": {
                name: _lag_one_autocorrelation(common, values)
                for name, values in common_targets.items()
            },
            "warning": (
                "Overlapping multi-hour targets mechanically introduce serial dependence; "
                "these correlations are not evidence of forecastability."
            ),
        },
        "right_edge_boundary_audit": {
            "boundary_crossing_exclusions_by_horizon": {
                f"{horizon}h": count
                for horizon, count in zip(
                    FROZEN_HORIZONS_HOURS,
                    right_edge_counts,
                    strict=True,
                )
            },
            "expected_boundary_crossing_exclusions_by_horizon": {
                f"{horizon}h": count
                for horizon, count in zip(
                    FROZEN_HORIZONS_HOURS,
                    expected_right_edge_counts,
                    strict=True,
                )
            },
            "monotonic_additional_purge_with_horizon": True,
            "target_endpoint_enters_original_validation_count": 0,
        },
        "one_hour_regression_check": regression,
        "model_training": False,
    }
    output = root / AUDIT_OUTPUT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return HorizonAuditResult(output, summary)
