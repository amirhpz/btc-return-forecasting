from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_generalization import (
    AUTOCORRELATION_LAGS_HOURS,
    build_generalization_report,
    build_target_regime_report,
    feature_target_stability,
    gap_safe_target_autocorrelation,
    target_regime_statistics,
    validation_block_statistics,
)


def _samples(
    targets: np.ndarray,
    *,
    start: str,
    timestamps: pd.DatetimeIndex | None = None,
) -> SequenceSamples:
    values = np.asarray(targets, dtype=np.float64)
    decision_times = (
        timestamps
        if timestamps is not None
        else pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    )
    features = np.zeros((len(values), 24, len(F0_FEATURE_NAMES)), dtype=np.float32)
    for index in range(len(F0_FEATURE_NAMES)):
        features[:, -1, index] = values * (1.0 if index % 2 == 0 else -1.0)
    return SequenceSamples(
        features=features,
        targets=values,
        decision_times=decision_times,
        candidate_count=len(values),
        excluded_lookback_count=0,
    )


def test_target_regime_and_validation_over_train_ratios_are_exact() -> None:
    train = np.array([-2.0, -1.0, 1.0, 2.0])
    validation = np.array([-1.0, 0.0, 1.0, 4.0])

    train_stats = target_regime_statistics(train)
    report = build_target_regime_report(train, validation)

    assert train_stats["n"] == 4
    assert train_stats["mean"] == float(np.mean(train))
    assert train_stats["std"] == float(np.std(train, ddof=0))
    assert train_stats["median"] == float(np.median(train))
    assert train_stats["mean_absolute_target"] == float(np.mean(np.abs(train)))
    assert train_stats["positive_ratio"] == 0.5
    assert train_stats["quantile_05"] == pytest.approx(np.quantile(train, 0.05))
    assert train_stats["quantile_25"] == pytest.approx(np.quantile(train, 0.25))
    assert train_stats["quantile_75"] == pytest.approx(np.quantile(train, 0.75))
    assert train_stats["quantile_95"] == pytest.approx(np.quantile(train, 0.95))
    assert report["validation_over_train"]["target_std"] == pytest.approx(
        np.std(validation, ddof=0) / np.std(train, ddof=0)
    )
    assert report["validation_over_train"]["mean_absolute_target"] == pytest.approx(
        np.mean(np.abs(validation)) / np.mean(np.abs(train))
    )


def test_validation_is_divided_into_exactly_four_contiguous_fixed_blocks() -> None:
    targets = np.linspace(-0.2, 0.3, 10)
    validation = _samples(targets, start="2026-01-01")
    predictions = targets * 0.7 + 0.01

    blocks = validation_block_statistics(validation, predictions)

    assert len(blocks) == 4
    assert [block["n"] for block in blocks] == [3, 3, 2, 2]
    assert blocks[0]["start_decision_time"] == validation.decision_times[0].isoformat()
    assert blocks[0]["end_decision_time"] == validation.decision_times[2].isoformat()
    assert blocks[-1]["start_decision_time"] == validation.decision_times[8].isoformat()
    assert blocks[-1]["end_decision_time"] == validation.decision_times[9].isoformat()
    assert sum(int(block["n"]) for block in blocks) == len(validation.targets)


def test_all_f0_endpoint_correlations_report_train_validation_sign_stability() -> None:
    targets = np.linspace(-1.0, 1.0, 20)
    train = _samples(targets, start="2025-01-01")
    validation = _samples(targets, start="2026-01-01")
    validation.features[:, -1, 1] *= -1.0

    result = feature_target_stability(train, validation)

    assert tuple(result) == F0_FEATURE_NAMES
    assert len(result) == 10
    assert result[F0_FEATURE_NAMES[0]]["pearson"]["sign_consistent"] is True
    assert result[F0_FEATURE_NAMES[0]]["spearman"]["sign_consistent"] is True
    assert result[F0_FEATURE_NAMES[1]]["pearson"]["sign_consistent"] is False
    assert result[F0_FEATURE_NAMES[1]]["spearman"]["sign_consistent"] is False


def test_return_autocorrelation_uses_only_exact_real_hour_pairs() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    retained_hours = [0, 1, 2, 4, 5, 8]
    timestamps = pd.DatetimeIndex(
        [start + timedelta(hours=hour) for hour in retained_hours]
    )
    samples = _samples(
        np.array([-0.3, 0.1, 0.2, -0.1, 0.4, 0.05]),
        start="2026-01-01",
        timestamps=timestamps,
    )

    result = gap_safe_target_autocorrelation(samples)

    assert tuple(int(lag) for lag in result) == AUTOCORRELATION_LAGS_HOURS
    assert result["1"]["n_pairs"] == 3
    assert result["3"]["n_pairs"] == 3
    assert result["6"]["n_pairs"] == 1
    assert result["12"]["n_pairs"] == 0
    assert result["24"]["n_pairs"] == 0
    assert result["6"]["autocorrelation"] is None


def test_generalization_report_is_train_validation_only_and_contains_all_sections() -> None:
    train_targets = np.linspace(-0.3, 0.3, 24)
    validation_targets = np.linspace(-0.2, 0.25, 12)
    train = _samples(train_targets, start="2025-01-01")
    validation = _samples(validation_targets, start="2026-01-01")

    result = build_generalization_report(
        train=train,
        validation=validation,
        validation_predictions=validation_targets * 0.5,
        source_run_id="E03_fixture",
        device="cuda",
        gpu_name="fixture_gpu",
    )

    assert result["evaluated_splits"] == ["train", "validation"]
    assert set(result) >= {
        "target_regime",
        "validation_temporal_blocks",
        "f0_endpoint_feature_target_stability",
        "target_return_autocorrelation",
    }
    assert len(result["validation_temporal_blocks"]) == 4
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_generalization_cli_requires_the_official_source_run() -> None:
    source = Path("outputs/runs/E03_fixture")
    args = build_parser().parse_args(
        ["diagnose-lstm-generalization", "--source-run", str(source)]
    )

    assert args.command == "diagnose-lstm-generalization"
    assert args.source_run == source
