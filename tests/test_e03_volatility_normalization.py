from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
    build_distribution_comparison,
    build_volatility_normalization_report,
    select_normalization_samples,
    validation_normalization_blocks,
    volatility_target_relationship,
)


def _samples(
    targets: np.ndarray,
    sigma: np.ndarray,
    *,
    start: str,
) -> SequenceSamples:
    target_values = np.asarray(targets, dtype=np.float64)
    sigma_values = np.asarray(sigma, dtype=np.float64)
    if target_values.shape != sigma_values.shape:
        raise ValueError("targets and sigma must have the same shape")
    features = np.zeros(
        (len(target_values), 24, len(F0_FEATURE_NAMES)),
        dtype=np.float64,
    )
    features[:, -1, VOLATILITY_FEATURE_INDEX] = sigma_values
    return SequenceSamples(
        features=features,
        targets=target_values,
        decision_times=pd.date_range(start, periods=len(target_values), freq="h", tz="UTC"),
        candidate_count=len(target_values),
        excluded_lookback_count=0,
    )


def test_zero_missing_and_nonfinite_sigma_are_excluded_without_epsilon() -> None:
    samples = _samples(
        np.array([1.0, 2.0, 3.0, 4.0, 6.0]),
        np.array([1.0, 0.0, np.nan, np.inf, 2.0]),
        start="2026-01-01",
    )

    selected = select_normalization_samples(samples)

    assert selected.exclusion_count == 3
    assert selected.valid_mask.tolist() == [True, False, False, False, True]
    assert selected.eligible_targets.tolist() == [1.0, 6.0]
    assert selected.eligible_sigma.tolist() == [1.0, 2.0]
    assert selected.normalized_targets.tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match="cannot be negative"):
        select_normalization_samples(
            _samples(np.array([1.0]), np.array([-0.1]), start="2026-01-01")
        )


def test_volatility_relationship_correlations_use_only_normalizable_samples() -> None:
    samples = _samples(
        np.array([-1.0, 2.0, -4.0, 8.0, 99.0]),
        np.array([1.0, 2.0, 4.0, 8.0, 0.0]),
        start="2026-01-01",
    )

    result = volatility_target_relationship(select_normalization_samples(samples))

    assert result["n"] == 4
    assert result["pearson_sigma_vs_abs_target"] == pytest.approx(1.0)
    assert result["spearman_sigma_vs_abs_target"] == pytest.approx(1.0)


def test_raw_and_normalized_train_validation_distributions_and_ratios_are_exact() -> None:
    train = select_normalization_samples(
        _samples(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 1.0, 2.0]),
            start="2025-01-01",
        )
    )
    validation = select_normalization_samples(
        _samples(
            np.array([2.0, 4.0, 6.0, 8.0]),
            np.array([2.0, 2.0, 3.0, 4.0]),
            start="2026-01-01",
        )
    )

    raw, normalized = build_distribution_comparison(train, validation)

    assert raw["train"]["n"] == raw["validation"]["n"] == 4
    assert raw["train"]["q05"] == pytest.approx(np.quantile(train.eligible_targets, 0.05))
    assert raw["validation_over_train"]["std"] == pytest.approx(
        np.std(validation.eligible_targets) / np.std(train.eligible_targets)
    )
    assert raw["validation_over_train"]["mean_absolute_target"] == pytest.approx(
        np.mean(np.abs(validation.eligible_targets))
        / np.mean(np.abs(train.eligible_targets))
    )
    assert normalized["train"]["median"] == pytest.approx(
        np.median(train.normalized_targets)
    )
    assert normalized["validation_over_train"]["std"] == pytest.approx(
        np.std(validation.normalized_targets) / np.std(train.normalized_targets)
    )
    assert normalized["validation_over_train"]["mean_absolute_value"] == pytest.approx(
        np.mean(np.abs(validation.normalized_targets))
        / np.mean(np.abs(train.normalized_targets))
    )


def test_validation_uses_same_four_blocks_before_sigma_exclusions() -> None:
    targets = np.arange(1.0, 11.0)
    sigma = np.ones(10)
    sigma[[1, 8]] = 0.0
    validation = select_normalization_samples(
        _samples(targets, sigma, start="2026-01-01")
    )

    blocks = validation_normalization_blocks(validation)

    assert len(blocks) == 4
    assert [block["original_e03_n"] for block in blocks] == [3, 3, 2, 2]
    assert [block["excluded_sigma_count"] for block in blocks] == [1, 0, 0, 1]
    assert [block["n"] for block in blocks] == [2, 3, 2, 1]
    assert blocks[0]["start_decision_time"] == validation.decision_times[0].isoformat()
    assert blocks[0]["end_decision_time"] == validation.decision_times[2].isoformat()
    assert blocks[-1]["start_decision_time"] == validation.decision_times[8].isoformat()
    assert blocks[-1]["end_decision_time"] == validation.decision_times[9].isoformat()


def test_report_is_train_validation_only_and_preserves_official_targets() -> None:
    train = _samples(
        np.linspace(-0.2, 0.2, 12),
        np.linspace(0.01, 0.03, 12),
        start="2025-01-01",
    )
    validation = _samples(
        np.linspace(-0.1, 0.15, 12),
        np.linspace(0.015, 0.025, 12),
        start="2026-01-01",
    )
    train_targets_before = train.targets.copy()
    validation_targets_before = validation.targets.copy()

    result = build_volatility_normalization_report(
        train=train,
        validation=validation,
        source_run_id="E03_fixture",
    )

    assert result["evaluated_splits"] == ["train", "validation"]
    assert result["definition"]["epsilon_added"] is False
    assert result["normalization_exclusions"] == {"train": 0, "validation": 0}
    assert len(result["validation_temporal_blocks"]) == 4
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result
    assert np.array_equal(train.targets, train_targets_before)
    assert np.array_equal(validation.targets, validation_targets_before)


def test_volatility_normalization_cli_requires_source_e03_run() -> None:
    source = Path("outputs/runs/E03_fixture")
    args = build_parser().parse_args(
        ["diagnose-lstm-volatility-normalization", "--source-run", str(source)]
    )

    assert args.command == "diagnose-lstm-volatility-normalization"
    assert args.source_run == source
