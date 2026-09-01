from __future__ import annotations

import inspect

import numpy as np
import pytest

from btc_forecasting.training.lstm_vn_multiseed import (
    FROZEN_SEEDS,
    aggregate_seed_metrics,
    build_multiseed_report,
    stability_gate,
    verify_seed_sample_identities,
)


def _seed_result(
    seed: int,
    *,
    value: float,
    train_identity: str = "train-identity",
    validation_identity: str = "validation-identity",
) -> dict[str, object]:
    return {
        "seed": seed,
        "train_sample_count": 100,
        "validation_sample_count": 20,
        "train_sample_identity_sha256": train_identity,
        "validation_sample_identity_sha256": validation_identity,
        "raw_return_validation": {
            "n": 20,
            "mae": 1.0 + value,
            "rmse": 2.0 + value,
            "r2": value,
            "pearson_ic": value,
            "spearman_rank_ic": value,
            "directional_accuracy": 0.50 + value,
        },
        "same_row_zero_return": {"n": 20, "mae": 2.0, "rmse": 3.0},
        "skill": {"mae": value, "rmse": value},
        "normalized_space": {
            "mse": 3.0 + value,
            "pearson_ic": value,
            "spearman_rank_ic": value,
            "prediction_std": 0.5 + value,
            "target_std": 1.0,
        },
        "training": {
            "best_epoch": 2,
            "epochs_trained": 5,
            "duration_seconds": 1.0,
        },
        "evaluated_splits": ["validation"],
        "test_set": "NOT EVALUATED",
    }


def _results(values: list[float]) -> list[dict[str, object]]:
    return [
        _seed_result(seed, value=value)
        for seed, value in zip(FROZEN_SEEDS, values, strict=True)
    ]


def test_exact_five_frozen_seeds() -> None:
    assert FROZEN_SEEDS == (42, 137, 271, 811, 2027)


def test_seed_sample_identities_must_match() -> None:
    results = _results([0.01] * 5)
    samples = verify_seed_sample_identities(results)

    assert samples == {
        "train_sample_count": 100,
        "validation_sample_count": 20,
        "train_sample_identity_sha256": "train-identity",
        "validation_sample_identity_sha256": "validation-identity",
        "identical_across_all_five_seeds": True,
    }

    results[-1]["validation_sample_identity_sha256"] = "different"
    with pytest.raises(ValueError, match="identical TRAIN and validation"):
        verify_seed_sample_identities(results)


def test_aggregate_statistics_use_all_five_seeds() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.05]
    aggregates = aggregate_seed_metrics(_results(values))

    assert aggregates["mae_skill"] == {
        "mean": pytest.approx(0.03),
        "median": pytest.approx(0.03),
        "standard_deviation": pytest.approx(float(np.std(values, ddof=0))),
        "minimum": pytest.approx(0.01),
        "maximum": pytest.approx(0.05),
    }
    assert aggregates["normalized_target_std"]["standard_deviation"] == 0.0


def test_prefrozen_stability_gate_is_exact() -> None:
    stable_results = _results([0.01, 0.02, 0.03, 0.04, -0.001])
    stable_aggregates = aggregate_seed_metrics(stable_results)
    counts, stable = stability_gate(stable_results, stable_aggregates)

    assert all(count == 4 for count in counts.values())
    assert stable is True

    unstable_results = _results([0.01, 0.02, 0.03, -0.20, -0.20])
    unstable_aggregates = aggregate_seed_metrics(unstable_results)
    _, unstable = stability_gate(unstable_results, unstable_aggregates)
    assert unstable is False


def test_report_has_no_test_input_or_evaluation() -> None:
    assert tuple(inspect.signature(build_multiseed_report).parameters) == (
        "seed_results",
    )
    report = build_multiseed_report(_results([0.01] * 5))

    assert report["evaluated_splits"] == ["validation"]
    assert report["test_set"] == "NOT EVALUATED"
    assert all(
        result["test_set"] == "NOT EVALUATED"  # type: ignore[index]
        for result in report["per_seed"]  # type: ignore[union-attr]
    )
