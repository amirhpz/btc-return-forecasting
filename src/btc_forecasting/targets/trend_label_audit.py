from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from btc_forecasting.data.canonical_1h import CANONICAL_1H_RELATIVE_PATH
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.targets.horizon_audit import _sample_identity
from btc_forecasting.training.lstm import _write_json
from btc_forecasting.training.lstm_horizon_walkforward import (
    EXPECTED_COMMON_COUNT,
    EXPECTED_COMMON_IDENTITY,
    PairedHorizonSamples,
    build_paired_temporal_design,
    load_paired_horizon_samples,
)
from btc_forecasting.training.volatility_normalization import VOLATILITY_FEATURE_NAME

AUDIT_ID = "E05-L-A"
AUDIT_OUTPUT_RELATIVE_PATH = Path("outputs/data/trend_label_audit/summary.json")
FROZEN_TIMEFRAME = "1h"
FROZEN_HORIZON_HOURS = 1
ATR_PERIOD = 14
NEUTRAL_QUANTILE = 1.0 / 3.0
CONFIRMATION_LENGTH = 3
ONE_HOUR = timedelta(hours=1)
DOWN = -1
NEUTRAL = 0
UP = 1
LABEL_VALUES = (DOWN, NEUTRAL, UP)
METHODS = ("L1_FIXED", "L2_ATR_ADAPTIVE", "L3_ATR_HYSTERESIS_3")
REGIMES = ("LOW", "MID", "HIGH")


@dataclass(frozen=True)
class HysteresisResult:
    labels: np.ndarray
    rejected_candidate_switches: int
    confirmed_switches: int


@dataclass(frozen=True)
class TrendLabelAuditResult:
    artifact_path: Path
    summary: dict[str, object]


def causal_wilder_atr14(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open_time", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ATR14 source is missing columns: {missing}")
    times = pd.DatetimeIndex(pd.to_datetime(frame["open_time"], utc=True))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("ATR14 timestamps must be strictly ordered and unique")
    high = frame["high"].to_numpy(np.float64)
    low = frame["low"].to_numpy(np.float64)
    close = frame["close"].to_numpy(np.float64)
    if (
        np.any(~np.isfinite(high))
        or np.any(~np.isfinite(low))
        or np.any(~np.isfinite(close))
        or np.any(high < low)
        or np.any(close <= 0.0)
    ):
        raise ValueError("ATR14 requires finite valid OHLC values and positive closes")

    true_range = np.full(len(frame), np.nan, dtype=np.float64)
    atr = np.full(len(frame), np.nan, dtype=np.float64)
    segment_start = 0
    for position in range(len(frame)):
        starts_segment = position == 0 or times[position] - times[position - 1] != ONE_HOUR
        if starts_segment:
            segment_start = position
            true_range[position] = high[position] - low[position]
        else:
            previous_close = close[position - 1]
            true_range[position] = max(
                high[position] - low[position],
                abs(high[position] - previous_close),
                abs(low[position] - previous_close),
            )
        segment_length = position - segment_start + 1
        if segment_length == ATR_PERIOD:
            atr[position] = float(np.mean(true_range[segment_start : position + 1]))
        elif segment_length > ATR_PERIOD:
            atr[position] = (
                atr[position - 1] * (ATR_PERIOD - 1) + true_range[position]
            ) / ATR_PERIOD
    atr_pct = atr / close
    return pd.DataFrame(
        {
            "open_time": times,
            "true_range": true_range,
            "atr_14": atr,
            "atr_pct_14": atr_pct,
        }
    )


def calibrate_fixed_tau(future_returns: np.ndarray) -> float:
    values = _finite_nonempty(future_returns, name="calibration future returns")
    tau = float(np.quantile(np.abs(values), NEUTRAL_QUANTILE, method="linear"))
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("Calibrated tau must be finite and non-negative")
    return tau


def calibrate_adaptive_k(
    future_returns: np.ndarray,
    atr_pct_14: np.ndarray,
) -> float:
    values = _finite_nonempty(future_returns, name="calibration future returns")
    atr = _finite_nonempty(atr_pct_14, name="calibration ATR percentage")
    if len(values) != len(atr) or np.any(atr <= 0.0):
        raise ValueError("Adaptive calibration requires aligned positive ATR percentages")
    ratio = np.abs(values) / atr
    k = float(np.quantile(ratio, NEUTRAL_QUANTILE, method="linear"))
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("Calibrated k must be finite and non-negative")
    return k


def calibrate_volatility_regimes(volatility: np.ndarray) -> tuple[float, float]:
    values = _finite_nonempty(volatility, name="calibration rolling volatility")
    low, high = np.quantile(values, [0.33, 0.67], method="linear")
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("Volatility regime cut points must be finite and ordered")
    return float(low), float(high)


def fixed_three_class_labels(future_returns: np.ndarray, *, tau: float) -> np.ndarray:
    values = _finite_nonempty(future_returns, name="future returns")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("Fixed threshold tau must be finite and non-negative")
    return np.where(values > tau, UP, np.where(values < -tau, DOWN, NEUTRAL)).astype(
        np.int8
    )


def adaptive_three_class_labels(
    future_returns: np.ndarray,
    atr_pct_14: np.ndarray,
    *,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_nonempty(future_returns, name="future returns")
    atr = _finite_nonempty(atr_pct_14, name="ATR percentage")
    if len(values) != len(atr) or np.any(atr <= 0.0):
        raise ValueError("Adaptive labeling requires aligned positive ATR percentages")
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("Adaptive multiplier k must be finite and non-negative")
    theta = k * atr
    labels = np.where(
        values > theta,
        UP,
        np.where(values < -theta, DOWN, NEUTRAL),
    ).astype(np.int8)
    return labels, theta


def hysteresis_three(
    candidate_labels: np.ndarray,
    decision_times: pd.DatetimeIndex,
) -> HysteresisResult:
    candidates = _valid_labels(candidate_labels)
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    if len(candidates) != len(times):
        raise ValueError("Hysteresis labels and timestamps must align")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Hysteresis timestamps must be strictly ordered and unique")
    output = np.full(len(candidates), NEUTRAL, dtype=np.int8)
    state = NEUTRAL
    pending_state: int | None = None
    pending_count = 0
    rejected = 0
    confirmed = 0
    for position, candidate_value in enumerate(candidates):
        candidate = int(candidate_value)
        if position == 0 or times[position] - times[position - 1] != ONE_HOUR:
            state = NEUTRAL
            pending_state = None
            pending_count = 0
        if candidate == state:
            pending_state = None
            pending_count = 0
        else:
            if pending_state == candidate:
                pending_count += 1
            else:
                pending_state = candidate
                pending_count = 1
            if pending_count == CONFIRMATION_LENGTH:
                state = candidate
                pending_state = None
                pending_count = 0
                confirmed += 1
            else:
                rejected += 1
        output[position] = state
    return HysteresisResult(output, rejected, confirmed)


def assign_volatility_regimes(
    volatility: np.ndarray,
    *,
    low_cut: float,
    high_cut: float,
) -> np.ndarray:
    values = _finite_nonempty(volatility, name="rolling volatility")
    if not np.isfinite(low_cut) or not np.isfinite(high_cut) or low_cut > high_cut:
        raise ValueError("Volatility cut points must be finite and ordered")
    return np.where(values <= low_cut, "LOW", np.where(values <= high_cut, "MID", "HIGH"))


def class_structure(labels: np.ndarray) -> dict[str, object]:
    values = _valid_labels(labels)
    count = len(values)
    counts = {label: int(np.count_nonzero(values == label)) for label in LABEL_VALUES}
    shares = {label: counts[label] / count for label in LABEL_VALUES}
    positive_shares = [share for share in shares.values() if share > 0.0]
    entropy = -sum(share * math.log(share) for share in positive_shares) / math.log(3.0)
    return {
        "sample_count": count,
        "down": {"count": counts[DOWN], "share": shares[DOWN]},
        "neutral": {"count": counts[NEUTRAL], "share": shares[NEUTRAL]},
        "up": {"count": counts[UP], "share": shares[UP]},
        "majority_class_share": max(shares.values()),
        "normalized_class_entropy": entropy,
    }


def temporal_behavior(
    labels: np.ndarray,
    decision_times: pd.DatetimeIndex,
) -> dict[str, float | int]:
    values = _valid_labels(labels)
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    if len(values) != len(times):
        raise ValueError("Temporal label diagnostics require aligned timestamps")
    transitions = 0
    direct_down_up = 0
    direct_up_down = 0
    durations: list[int] = []
    duration = 1
    for position in range(1, len(values)):
        consecutive = times[position] - times[position - 1] == ONE_HOUR
        if not consecutive:
            durations.append(duration)
            duration = 1
            continue
        if values[position] == values[position - 1]:
            duration += 1
            continue
        durations.append(duration)
        duration = 1
        transitions += 1
        direct_down_up += int(values[position - 1] == DOWN and values[position] == UP)
        direct_up_down += int(values[position - 1] == UP and values[position] == DOWN)
    durations.append(duration)
    array = np.asarray(durations, dtype=np.float64)
    return {
        "total_label_transitions": transitions,
        "transitions_per_1000_valid_hours": 1000.0 * transitions / len(values),
        "direct_down_to_up_transitions": direct_down_up,
        "direct_up_to_down_transitions": direct_up_down,
        "mean_state_duration_hours": float(np.mean(array)),
        "median_state_duration_hours": float(np.median(array)),
        "p90_state_duration_hours": float(np.quantile(array, 0.90, method="linear")),
        "maximum_state_duration_hours": int(np.max(array)),
    }


def semantic_integrity(labels: np.ndarray, future_returns: np.ndarray) -> dict[str, object]:
    values = _valid_labels(labels)
    returns = _finite_nonempty(future_returns, name="semantic future returns")
    if len(values) != len(returns):
        raise ValueError("Semantic diagnostics require aligned labels and returns")

    def directional(label: int) -> dict[str, float | int | None]:
        selected = returns[values == label]
        if len(selected) == 0:
            return {
                "count": 0,
                "mean_future_return": None,
                "median_future_return": None,
                "fraction_positive": None,
                "fraction_negative": None,
            }
        return {
            "count": len(selected),
            "mean_future_return": float(np.mean(selected)),
            "median_future_return": float(np.median(selected)),
            "fraction_positive": float(np.mean(selected > 0.0)),
            "fraction_negative": float(np.mean(selected < 0.0)),
        }

    neutral = np.abs(returns[values == NEUTRAL])
    non_neutral = values != NEUTRAL
    sign_agreement = (
        None
        if not np.any(non_neutral)
        else float(np.mean(np.sign(returns[non_neutral]) == values[non_neutral]))
    )
    return {
        "up": directional(UP),
        "down": directional(DOWN),
        "neutral": {
            "count": len(neutral),
            "mean_absolute_future_return": (
                None if len(neutral) == 0 else float(np.mean(neutral))
            ),
            "median_absolute_future_return": (
                None if len(neutral) == 0 else float(np.median(neutral))
            ),
        },
        "non_neutral_sign_agreement": sign_agreement,
    }


def signal_activity(
    labels: np.ndarray,
    decision_times: pd.DatetimeIndex,
) -> dict[str, int]:
    values = _valid_labels(labels)
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    temporal = temporal_behavior(values, times)
    entries = 0
    exits = 0
    previous_long = False
    for position, label in enumerate(values):
        if position == 0 or times[position] - times[position - 1] != ONE_HOUR:
            previous_long = False
        current_long = label == UP
        entries += int(current_long and not previous_long)
        exits += int(previous_long and not current_long)
        previous_long = current_long
    return {
        "three_state_signal_changes": int(temporal["total_label_transitions"]),
        "spot_style_long_entries": entries,
        "spot_style_long_exits": exits,
    }


def feasibility_decision(
    evaluation_class_structures: list[dict[str, object]],
) -> dict[str, object]:
    if len(evaluation_class_structures) != 4:
        raise ValueError("Feasibility requires exactly four outer folds")
    names = ("down", "neutral", "up")
    below = {
        name: sum(float(fold[name]["share"]) < 0.10 for fold in evaluation_class_structures)  # type: ignore[index]
        for name in names
    }
    above = {
        name: sum(float(fold[name]["share"]) > 0.80 for fold in evaluation_class_structures)  # type: ignore[index]
        for name in names
    }
    pathological = any(count >= 2 for count in (*below.values(), *above.values()))
    return {
        "result": "PATHOLOGICAL" if pathological else "FEASIBLE",
        "folds_below_10_percent_by_class": below,
        "folds_above_80_percent_by_class": above,
        "rule": (
            "pathological iff any class is below 10% in at least two folds or "
            "above 80% in at least two folds"
        ),
    }


def _finite_nonempty(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty one-dimensional array")
    return array


def _valid_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or len(values) == 0 or not set(np.unique(values)).issubset(
        LABEL_VALUES
    ):
        raise ValueError("Labels must be a non-empty vector containing only -1, 0, +1")
    return values


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = _finite_nonempty(values, name="threshold distribution")
    quantiles = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95], method="linear")
    return {
        "n": len(array),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "maximum": float(np.max(array)),
    }


def _regime_diagnostics(
    labels: np.ndarray,
    returns: np.ndarray,
    times: pd.DatetimeIndex,
    regimes: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for regime in REGIMES:
        selected = np.flatnonzero(regimes == regime)
        temporal = temporal_behavior(labels[selected], times[selected])
        result[regime] = {
            "class_distribution": class_structure(labels[selected]),
            "transition_rate_per_1000_hours": temporal[
                "transitions_per_1000_valid_hours"
            ],
            "mean_state_duration_hours": temporal["mean_state_duration_hours"],
            "mean_absolute_future_return": float(np.mean(np.abs(returns[selected]))),
        }
    return result


def _method_diagnostics(
    *,
    labels: np.ndarray,
    returns: np.ndarray,
    times: pd.DatetimeIndex,
    regimes: np.ndarray | None,
    theta: np.ndarray | None,
    hysteresis: HysteresisResult | None,
    l2_temporal: dict[str, float | int] | None,
) -> dict[str, object]:
    temporal = temporal_behavior(labels, times)
    result: dict[str, object] = {
        "class_structure": class_structure(labels),
        "temporal_behavior": temporal,
        "semantic_integrity": semantic_integrity(labels, returns),
        "implied_signal_activity": signal_activity(labels, times),
    }
    if theta is not None:
        result["adaptive_threshold_distribution"] = _distribution(theta)
    if regimes is not None:
        result["volatility_regimes"] = _regime_diagnostics(
            labels,
            returns,
            times,
            regimes,
        )
    if hysteresis is not None:
        if l2_temporal is None:
            raise ValueError("L3 diagnostics require paired L2 temporal diagnostics")
        l2_transitions = int(l2_temporal["total_label_transitions"])
        l3_transitions = int(temporal["total_label_transitions"])
        l2_duration = float(l2_temporal["mean_state_duration_hours"])
        l3_duration = float(temporal["mean_state_duration_hours"])
        result["hysteresis"] = {
            "candidate_switches_rejected": hysteresis.rejected_candidate_switches,
            "confirmed_switches": hysteresis.confirmed_switches,
            "transition_reduction_count_vs_l2": l2_transitions - l3_transitions,
            "transition_reduction_fraction_vs_l2": (
                None
                if l2_transitions == 0
                else 1.0 - l3_transitions / l2_transitions
            ),
            "mean_duration_increase_hours_vs_l2": l3_duration - l2_duration,
            "mean_duration_ratio_vs_l2": (
                None if l2_duration == 0.0 else l3_duration / l2_duration
            ),
        }
    return result


def audit_fold(
    *,
    fold_number: int,
    samples: PairedHorizonSamples,
    calibration_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    atr_pct_14: np.ndarray,
    endpoint_volatility: np.ndarray,
) -> dict[str, object]:
    calibration_y = samples.raw_targets[FROZEN_HORIZON_HOURS][calibration_positions]
    evaluation_y = samples.raw_targets[FROZEN_HORIZON_HOURS][evaluation_positions]
    calibration_atr = atr_pct_14[calibration_positions]
    evaluation_atr = atr_pct_14[evaluation_positions]
    calibration_volatility = endpoint_volatility[calibration_positions]
    evaluation_volatility = endpoint_volatility[evaluation_positions]
    tau = calibrate_fixed_tau(calibration_y)
    k = calibrate_adaptive_k(calibration_y, calibration_atr)
    regime_low, regime_high = calibrate_volatility_regimes(calibration_volatility)
    evaluation_regimes = assign_volatility_regimes(
        evaluation_volatility,
        low_cut=regime_low,
        high_cut=regime_high,
    )

    calibration_l1 = fixed_three_class_labels(calibration_y, tau=tau)
    evaluation_l1 = fixed_three_class_labels(evaluation_y, tau=tau)
    calibration_l2, calibration_theta = adaptive_three_class_labels(
        calibration_y,
        calibration_atr,
        k=k,
    )
    evaluation_l2, evaluation_theta = adaptive_three_class_labels(
        evaluation_y,
        evaluation_atr,
        k=k,
    )
    calibration_hysteresis = hysteresis_three(
        calibration_l2,
        samples.decision_times[calibration_positions],
    )
    evaluation_hysteresis = hysteresis_three(
        evaluation_l2,
        samples.decision_times[evaluation_positions],
    )
    calibration_times = samples.decision_times[calibration_positions]
    evaluation_times = samples.decision_times[evaluation_positions]
    calibration_l2_temporal = temporal_behavior(calibration_l2, calibration_times)
    evaluation_l2_temporal = temporal_behavior(evaluation_l2, evaluation_times)
    methods = {
        "L1_FIXED": {
            "calibration_train": _method_diagnostics(
                labels=calibration_l1,
                returns=calibration_y,
                times=calibration_times,
                regimes=None,
                theta=None,
                hysteresis=None,
                l2_temporal=None,
            ),
            "outer_evaluation": _method_diagnostics(
                labels=evaluation_l1,
                returns=evaluation_y,
                times=evaluation_times,
                regimes=evaluation_regimes,
                theta=None,
                hysteresis=None,
                l2_temporal=None,
            ),
        },
        "L2_ATR_ADAPTIVE": {
            "calibration_train": _method_diagnostics(
                labels=calibration_l2,
                returns=calibration_y,
                times=calibration_times,
                regimes=None,
                theta=calibration_theta,
                hysteresis=None,
                l2_temporal=None,
            ),
            "outer_evaluation": _method_diagnostics(
                labels=evaluation_l2,
                returns=evaluation_y,
                times=evaluation_times,
                regimes=evaluation_regimes,
                theta=evaluation_theta,
                hysteresis=None,
                l2_temporal=None,
            ),
        },
        "L3_ATR_HYSTERESIS_3": {
            "calibration_train": _method_diagnostics(
                labels=calibration_hysteresis.labels,
                returns=calibration_y,
                times=calibration_times,
                regimes=None,
                theta=calibration_theta,
                hysteresis=calibration_hysteresis,
                l2_temporal=calibration_l2_temporal,
            ),
            "outer_evaluation": _method_diagnostics(
                labels=evaluation_hysteresis.labels,
                returns=evaluation_y,
                times=evaluation_times,
                regimes=evaluation_regimes,
                theta=evaluation_theta,
                hysteresis=evaluation_hysteresis,
                l2_temporal=evaluation_l2_temporal,
            ),
        },
    }
    return {
        "fold": fold_number,
        "calibration_train": _scope_summary(samples, calibration_positions),
        "outer_evaluation": _scope_summary(samples, evaluation_positions),
        "calibration_parameters": {
            "tau": tau,
            "k": k,
            "neutral_quantile": NEUTRAL_QUANTILE,
            "volatility_regime_33rd_percentile": regime_low,
            "volatility_regime_67th_percentile": regime_high,
        },
        "methods": methods,
    }


def _scope_summary(
    samples: PairedHorizonSamples,
    positions: np.ndarray,
) -> dict[str, object]:
    times = samples.decision_times[positions]
    return {
        "sample_count": len(times),
        "first_decision_time": times[0].isoformat(),
        "last_decision_time": times[-1].isoformat(),
        "sample_identity_sha256": _sample_identity(times),
    }


def _atr_on_common_anchors(
    *,
    project_root: Path,
    decision_times: pd.DatetimeIndex,
) -> np.ndarray:
    anchor_times = decision_times - ONE_HOUR
    maximum_anchor = anchor_times[-1].to_pydatetime()
    canonical = pq.read_table(
        project_root / CANONICAL_1H_RELATIVE_PATH,
        columns=["open_time", "high", "low", "close"],
        filters=[("open_time", "<=", maximum_anchor)],
    ).to_pandas()
    atr = causal_wilder_atr14(canonical).set_index("open_time")
    values = atr.reindex(anchor_times)["atr_pct_14"].to_numpy(np.float64)
    if len(values) != len(decision_times) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Every COMMON anchor must have finite positive causal ATR percentage")
    return values


def _calibration_positions(
    block_positions: list[np.ndarray],
    evaluation_positions: np.ndarray,
    decision_times: pd.DatetimeIndex,
) -> tuple[np.ndarray, int]:
    unpurged = np.concatenate(block_positions)
    boundary = decision_times[evaluation_positions[0]]
    safe = unpurged[decision_times[unpurged] + ONE_HOUR < boundary]
    return safe, len(unpurged) - len(safe)


def _semantic_degradation(folds: list[dict[str, object]]) -> dict[str, object]:
    l2: list[float] = []
    l3: list[float] = []
    for fold in folds:
        methods = fold["methods"]
        for name, destination in (
            ("L2_ATR_ADAPTIVE", l2),
            ("L3_ATR_HYSTERESIS_3", l3),
        ):
            value = methods[name]["outer_evaluation"]["semantic_integrity"][  # type: ignore[index]
                "non_neutral_sign_agreement"
            ]
            if value is None:
                raise ValueError("Semantic degradation requires non-neutral labels in every fold")
            destination.append(float(value))
    l2_mean = float(np.mean(l2))
    l3_mean = float(np.mean(l3))
    return {
        "l2_mean_non_neutral_sign_agreement": l2_mean,
        "l3_mean_non_neutral_sign_agreement": l3_mean,
        "l3_minus_l2_percentage_points": 100.0 * (l3_mean - l2_mean),
        "SEMANTIC_DEGRADATION": l3_mean < l2_mean - 0.05,
        "rule": "true iff L3 mean agreement is more than 5 percentage points below L2",
    }


def run_trend_label_audit(*, project_root: Path) -> TrendLabelAuditResult:
    root = project_root.resolve()
    samples = load_paired_horizon_samples(project_root=root)
    if (
        len(samples.decision_times) != EXPECTED_COMMON_COUNT
        or _sample_identity(samples.decision_times) != EXPECTED_COMMON_IDENTITY
    ):
        raise ValueError("E05-L-A requires the exact E04-H-WF4 COMMON anchors")
    blocks, _ = build_paired_temporal_design(samples.decision_times)
    atr_pct = _atr_on_common_anchors(project_root=root, decision_times=samples.decision_times)
    volatility_index = F0_FEATURE_NAMES.index(VOLATILITY_FEATURE_NAME)
    endpoint_volatility = samples.features[:, -1, volatility_index].astype(np.float64)
    if np.any(~np.isfinite(endpoint_volatility)):
        raise ValueError("COMMON endpoint rolling volatility must be finite")

    fold_reports: list[dict[str, object]] = []
    for fold_number in range(1, 5):
        evaluation = blocks[fold_number + 1].positions
        calibration, purged = _calibration_positions(
            [block.positions for block in blocks[: fold_number + 1]],
            evaluation,
            samples.decision_times,
        )
        report = audit_fold(
            fold_number=fold_number,
            samples=samples,
            calibration_positions=calibration,
            evaluation_positions=evaluation,
            atr_pct_14=atr_pct,
            endpoint_volatility=endpoint_volatility,
        )
        report["target_boundary_purge_count"] = purged
        fold_reports.append(report)

    feasibility: dict[str, object] = {}
    for method in METHODS:
        structures = [
            fold["methods"][method]["outer_evaluation"]["class_structure"]  # type: ignore[index]
            for fold in fold_reports
        ]
        feasibility[method] = feasibility_decision(structures)
    semantic_degradation = _semantic_degradation(fold_reports)
    feasibility["L3_ATR_HYSTERESIS_3"]["SEMANTIC_DEGRADATION"] = (  # type: ignore[index]
        semantic_degradation["SEMANTIC_DEGRADATION"]
    )

    full_positions = np.arange(len(samples.decision_times))
    full_y = samples.raw_targets[FROZEN_HORIZON_HOURS]
    regime_low, regime_high = calibrate_volatility_regimes(endpoint_volatility)
    summary: dict[str, object] = {
        "audit_id": AUDIT_ID,
        "model_training": False,
        "frozen_problem": {
            "input_timeframe": FROZEN_TIMEFRAME,
            "forecast_horizon_hours": FROZEN_HORIZON_HOURS,
            "target_formula": "log(close[t+1h] / close[t])",
            "decision_time": "after completion of bar t",
            "source_scope": "frozen TRAIN COMMON anchors only",
            "common_anchor_count": len(samples.decision_times),
            "common_anchor_identity_sha256": _sample_identity(samples.decision_times),
        },
        "label_definitions": {
            "labels": {"DOWN": DOWN, "NEUTRAL": NEUTRAL, "UP": UP},
            "L1_FIXED": {
                "tau": "quantile_calibration_train(abs(y), 1/3)",
                "formula": "UP if y > tau; DOWN if y < -tau; NEUTRAL otherwise",
            },
            "L2_ATR_ADAPTIVE": {
                "k": "quantile_calibration_train(abs(y) / atr_pct_14, 1/3)",
                "theta_t": "k * atr_pct_14_t",
                "formula": "UP if y > theta_t; DOWN if y < -theta_t; NEUTRAL otherwise",
            },
            "L3_ATR_HYSTERESIS_3": {
                "source": "L2 candidate labels",
                "initial_state": "NEUTRAL",
                "confirmation_length": CONFIRMATION_LENGTH,
                "rule": "switch only after three consecutive identical candidates differing from current state",
                "gap_rule": "reset state and pending confirmation to NEUTRAL",
                "causal": True,
            },
        },
        "atr_implementation": {
            "period": ATR_PERIOD,
            "true_range": "max(high_t-low_t, abs(high_t-close_t-1), abs(low_t-close_t-1))",
            "segment_first_true_range": "high_t - low_t because no prior consecutive close exists",
            "initialization": "arithmetic mean of first 14 causal true ranges per uninterrupted segment",
            "recurrence": "ATR_t = (13 * ATR_t-1 + TR_t) / 14",
            "normalized": "atr_pct_14_t = ATR14_t / close_t",
            "gap_behavior": "ATR state resets; no filling or interpolation",
            "alignment": "bar-t ATR is available at decision time after bar t completes",
        },
        "calibration": {
            "neutral_quantile": NEUTRAL_QUANTILE,
            "quantile_method": "linear",
            "fold_parameters_use": "preceding leakage-purged calibration TRAIN only",
            "evaluation_recalibration": False,
        },
        "temporal_audit": {
            "block_count": 6,
            "fold_count": 4,
            "folds": fold_reports,
        },
        "feasibility": feasibility,
        "l3_semantic_degradation": semantic_degradation,
        "full_train_reference_parameters": {
            "scope": _scope_summary(samples, full_positions),
            "tau_full_train": calibrate_fixed_tau(full_y),
            "k_full_train": calibrate_adaptive_k(full_y, atr_pct),
            "volatility_regime_33rd_percentile": regime_low,
            "volatility_regime_67th_percentile": regime_high,
            "applied_to_original_validation_or_test": False,
        },
        "original_validation": "NOT READ OR USED",
        "test_set": "NOT READ OR USED",
        "classification_metrics": "NOT COMPUTED",
        "profitability_backtest": "NOT PERFORMED",
    }
    artifact_path = root / AUDIT_OUTPUT_RELATIVE_PATH
    temporary = artifact_path.with_suffix(".json.tmp")
    _write_json(temporary, summary)
    os.replace(temporary, artifact_path)
    return TrendLabelAuditResult(artifact_path=artifact_path, summary=summary)
