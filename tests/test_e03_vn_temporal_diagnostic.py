from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.cli import build_parser
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples
from btc_forecasting.training.lstm_vn_mse import (
    VolatilityNormalizedSamples,
    prepare_volatility_normalized_samples,
)
from btc_forecasting.training.lstm_vn_temporal_diagnostic import (
    TRAIN_BLOCK_COUNT,
    build_temporal_diagnostic_report,
    normalized_feature_signal_stability,
    summarize_training_history,
    train_temporal_block_statistics,
)
from btc_forecasting.training.volatility_normalization import (
    VOLATILITY_FEATURE_INDEX,
)


def _normalized_samples(
    normalized_targets: np.ndarray,
    *,
    start: str,
    reverse_non_volatility_features: bool = False,
) -> VolatilityNormalizedSamples:
    z = np.asarray(normalized_targets, dtype=np.float64)
    sigma = 1.0 + (z - np.min(z) + 0.1)
    features = np.zeros((len(z), 24, len(F0_FEATURE_NAMES)), dtype=np.float32)
    endpoint = -z if reverse_non_volatility_features else z
    features[:, -1, :] = endpoint[:, None]
    features[:, -1, VOLATILITY_FEATURE_INDEX] = sigma
    samples = SequenceSamples(
        features=features,
        targets=z * sigma,
        decision_times=pd.date_range(start, periods=len(z), freq="h", tz="UTC"),
        candidate_count=len(z),
        excluded_lookback_count=0,
    )
    return prepare_volatility_normalized_samples(samples)


def test_training_history_reports_only_completed_epoch_fields_and_summary() -> None:
    history = [
        {
            "epoch": 1,
            "train_mse_loss": 2.0,
            "validation_mse_loss": 1.5,
            "learning_rate": 0.001,
            "ignored": 99,
        },
        {
            "epoch": 2,
            "train_mse_loss": 1.8,
            "validation_mse_loss": 1.4,
            "learning_rate": 0.0009,
        },
        {
            "epoch": 3,
            "train_mse_loss": 1.6,
            "validation_mse_loss": 1.45,
            "learning_rate": 0.0008,
        },
    ]

    result = summarize_training_history(history)

    assert result["epochs"] == [
        {
            "epoch": 1,
            "train_normalized_target_mse": 2.0,
            "validation_normalized_target_mse": 1.5,
            "learning_rate": 0.001,
        },
        {
            "epoch": 2,
            "train_normalized_target_mse": 1.8,
            "validation_normalized_target_mse": 1.4,
            "learning_rate": 0.0009,
        },
        {
            "epoch": 3,
            "train_normalized_target_mse": 1.6,
            "validation_normalized_target_mse": 1.45,
            "learning_rate": 0.0008,
        },
    ]
    assert result["best_validation_epoch"] == 2
    assert result["lowest_train_mse_epoch"] == 3
    assert result["final_train_mse"] == 1.6
    assert result["final_validation_mse"] == 1.45


def test_train_temporal_stability_uses_exactly_six_contiguous_blocks() -> None:
    prepared = _normalized_samples(
        np.linspace(-1.0, 1.0, 13),
        start="2020-01-01",
    )
    predictions = prepared.normalized.targets * 0.5

    blocks = train_temporal_block_statistics(
        prepared.normalized,
        predictions,
    )

    assert TRAIN_BLOCK_COUNT == 6
    assert len(blocks) == 6
    assert [block["n"] for block in blocks] == [3, 2, 2, 2, 2, 2]
    assert blocks[0]["start_decision_time"] == prepared.raw.decision_times[0].isoformat()
    assert blocks[0]["end_decision_time"] == prepared.raw.decision_times[2].isoformat()
    assert blocks[-1]["start_decision_time"] == prepared.raw.decision_times[11].isoformat()
    assert blocks[-1]["end_decision_time"] == prepared.raw.decision_times[12].isoformat()
    assert blocks[0]["prediction_std_over_target_std"] == pytest.approx(0.5)
    assert blocks[0]["mse_skill"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="exactly six"):
        train_temporal_block_statistics(
            prepared.normalized,
            predictions,
            block_count=5,
        )


def test_all_ten_endpoint_features_report_normalized_target_spearman_stability() -> None:
    z = np.linspace(-0.4, 0.4, 12)
    train = _normalized_samples(z, start="2020-01-01")
    validation = _normalized_samples(
        z,
        start="2026-01-01",
        reverse_non_volatility_features=True,
    )

    result = normalized_feature_signal_stability(train, validation)

    assert tuple(result) == F0_FEATURE_NAMES
    assert len(result) == 10
    assert result["log_return_1h"]["train_spearman"] == pytest.approx(1.0)
    assert result["log_return_1h"]["validation_spearman"] == pytest.approx(-1.0)
    assert result["log_return_1h"]["sign_consistent"] is False
    assert result["rolling_volatility_24h"]["train_spearman"] == pytest.approx(1.0)
    assert result["rolling_volatility_24h"]["validation_spearman"] == pytest.approx(
        1.0
    )
    assert result["rolling_volatility_24h"]["sign_consistent"] is True


def test_temporal_report_is_read_only_train_validation_and_never_test() -> None:
    result = build_temporal_diagnostic_report(
        source_run_id="E03-VN-MSE_fixture",
        training_history={"epochs": [{"epoch": 1}]},
        train_blocks=[{"block": number} for number in range(1, 7)],
        feature_stability={name: {} for name in F0_FEATURE_NAMES},
        train_invalid_sigma_exclusions=0,
        validation_invalid_sigma_exclusions=0,
        device="cuda",
        gpu_name="fixture_gpu",
    )

    assert result["diagnostic_id"] == "E03-VN-TD"
    assert len(result["train_temporal_blocks"]) == 6
    assert len(result["normalized_target_endpoint_feature_signal_stability"]) == 10
    assert result["normalization_exclusions"] == {"train": 0, "validation": 0}
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_vn_temporal_cli_requires_one_source_run() -> None:
    source = Path("outputs/runs/E03-VN-MSE_fixture")
    args = build_parser().parse_args(
        ["diagnose-lstm-vn-temporal", "--source-run", str(source)]
    )

    assert args.command == "diagnose-lstm-vn-temporal"
    assert args.source_run == source
