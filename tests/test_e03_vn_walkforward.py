from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.targets.one_hour import TARGET_COLUMN
from btc_forecasting.training.lstm import SequenceSamples, TrainingOutcome, _sample_identity
from btc_forecasting.training.lstm_vn_mse import VolatilityNormalizedSamples
from btc_forecasting.training.lstm_vn_multiseed import FROZEN_SEEDS, PRIMARY_METRIC_PATHS
from btc_forecasting.training.lstm_vn_walkforward import (
    BLOCK_COUNT,
    FOLD_COUNT,
    FixedEpochOutcome,
    FoldScopes,
    _load_frozen_train_targets,
    aggregate_evaluations,
    build_temporal_design,
    selected_refit_epoch_count,
    temporal_stability_gate,
    train_fold_seed,
)


def _sequence(start: str, count: int, *, offset: float = 0.0) -> SequenceSamples:
    times = pd.date_range(start, periods=count, freq="h", tz="UTC")
    targets = np.linspace(-0.2, 0.2, count) + offset
    return SequenceSamples(
        features=np.full((count, 24, len(F0_FEATURE_NAMES)), offset, dtype=np.float32),
        targets=targets,
        decision_times=times,
        candidate_count=count,
        excluded_lookback_count=0,
    )


def _vn(start: str, count: int, *, offset: float = 0.0) -> VolatilityNormalizedSamples:
    normalized = _sequence(start, count, offset=offset)
    raw = SequenceSamples(
        features=normalized.features,
        targets=normalized.targets * 0.5,
        decision_times=normalized.decision_times,
        candidate_count=count,
        excluded_lookback_count=0,
    )
    return VolatilityNormalizedSamples(
        raw=raw,
        normalized=normalized,
        sigma=np.full(count, 0.5),
        exclusion_count=0,
        original_eligible_count=count,
    )


def test_six_blocks_four_expanding_folds_and_five_seeds() -> None:
    times = pd.date_range("2020-01-01", periods=61, freq="h", tz="UTC")
    blocks, folds = build_temporal_design(times)

    assert BLOCK_COUNT == 6
    assert FOLD_COUNT == 4
    assert FROZEN_SEEDS == (42, 137, 271, 811, 2027)
    assert len(blocks) == 6
    assert len(folds) == 4
    assert np.array_equal(np.concatenate([block.positions for block in blocks]), np.arange(61))
    assert max(len(block.positions) for block in blocks) - min(
        len(block.positions) for block in blocks
    ) == 1
    assert [fold.outer_evaluation_positions[0] for fold in folds] == [
        blocks[index].positions[0] for index in range(2, 6)
    ]
    assert all(
        len(folds[index].outer_pool_positions)
        < len(folds[index + 1].outer_pool_positions)
        for index in range(3)
    )


def test_inner_and_outer_boundaries_are_strictly_leakage_safe() -> None:
    times = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
    _, folds = build_temporal_design(times)

    for fold in folds:
        inner_start = times[fold.inner_validation_positions[0]]
        outer_start = times[fold.outer_evaluation_positions[0]]
        assert times[fold.inner_train_positions[-1]] + np.timedelta64(1, "h") < inner_start
        assert times[fold.outer_pool_positions[-1]] + np.timedelta64(1, "h") < outer_start
        assert times[fold.inner_validation_positions[-1]] < outer_start
        assert set(fold.inner_train_positions).isdisjoint(fold.inner_validation_positions)
        assert set(fold.outer_pool_positions).isdisjoint(fold.outer_evaluation_positions)


def test_outer_evaluation_is_not_used_for_early_stopping_or_scaler_fit(
    monkeypatch,
) -> None:
    scopes = FoldScopes(
        outer_pool=_vn("2020-01-01", 20, offset=0.1),
        inner_train=_vn("2020-01-01", 12, offset=0.2),
        inner_validation=_vn("2020-01-02", 4, offset=0.3),
        outer_evaluation=_vn("2020-02-01", 5, offset=0.4),
    )
    events: list[str] = []
    fitted_ids: list[int] = []

    def fake_fit_scope(samples: SequenceSamples):
        fitted_ids.append(id(samples))
        events.append("fit_scaler")
        return object(), samples

    def fake_transform(samples: SequenceSamples, scaler: object) -> SequenceSamples:
        if samples is scopes.outer_evaluation.normalized:
            events.append("transform_outer_evaluation")
        return samples

    def fake_fit_lstm(model, *, train, validation, training_config, seed, device):
        assert validation is scopes.inner_validation.normalized
        assert validation is not scopes.outer_evaluation.normalized
        events.append("inner_early_stopping")
        return TrainingOutcome(
            model=model,
            history=[{"epoch": 1}],
            best_epoch=2,
            epochs_trained=3,
            best_validation_loss=0.1,
            duration_seconds=1.0,
        )

    def fake_refit(model, *, train, training_config, seed, device, epochs):
        assert train is scopes.outer_pool.normalized
        assert epochs == 2
        events.append("outer_refit_complete")
        return FixedEpochOutcome(
            model=model,
            history=[{"epoch": 1}, {"epoch": 2}],
            duration_seconds=2.0,
        )

    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.configure_determinism",
        lambda seed: None,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward._new_model",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.fit_scope_scaler",
        fake_fit_scope,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.transform_scope",
        fake_transform,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.fit_lstm",
        fake_fit_lstm,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.fit_lstm_fixed_epochs",
        fake_refit,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.predict_lstm",
        lambda model, features, batch_size, device: np.linspace(-0.1, 0.1, len(features)),
    )

    trained = train_fold_seed(
        fold=1,
        seed=42,
        scopes=scopes,
        model_config={"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        training_config={"max_epochs": 30, "batch_size": 128},
        device=torch.device("cpu"),
    )

    assert fitted_ids == [
        id(scopes.inner_train.normalized),
        id(scopes.outer_pool.normalized),
    ]
    assert events.index("outer_refit_complete") < events.index(
        "transform_outer_evaluation"
    )
    assert trained.metrics["training"]["refit_epoch_count"] == 2  # type: ignore[index]


def test_refit_epoch_count_is_exactly_inner_selected_epoch() -> None:
    assert selected_refit_epoch_count(7, max_epochs=30) == 7
    with pytest.raises(ValueError):
        selected_refit_epoch_count(31, max_epochs=30)


def test_target_loader_stops_before_original_validation_and_test(monkeypatch) -> None:
    rows = pd.DataFrame(
        {
            "bar_open_time": [datetime(2020, 1, 1, tzinfo=UTC)],
            "decision_time": [datetime(2020, 1, 1, 1, tzinfo=UTC)],
            "target_time": [datetime(2020, 1, 1, 2, tzinfo=UTC)],
            TARGET_COLUMN: [0.01],
        }
    )
    captured: dict[str, object] = {}

    class FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return rows.copy()

    def fake_read_table(path: Path, *, columns: list[str], filters: list[tuple]) -> FakeTable:
        captured["filters"] = filters
        return FakeTable()

    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_walkforward.pq.read_table",
        fake_read_table,
    )
    metadata = {
        "split": {
            "retained_rows": {"train": 1},
            "boundaries": {
                "train": {
                    "decision_time_start_inclusive": "2020-01-01T00:00:00Z",
                    "target_time_end_exclusive": "2020-02-01T00:00:00Z",
                },
                "validation": {
                    "decision_time_start_inclusive": "2020-02-01T00:00:00Z",
                },
            },
        }
    }

    result = _load_frozen_train_targets(
        Path("targets.parquet"),
        split_metadata=metadata,
    )

    assert len(result) == 1
    assert captured["filters"] == [
        ("decision_time", ">=", datetime(2020, 1, 1, tzinfo=UTC)),
        ("decision_time", "<", datetime(2020, 2, 1, tzinfo=UTC)),
    ]


def test_sample_identities_are_deterministic() -> None:
    times = pd.date_range("2020-01-01", periods=60, freq="h", tz="UTC")
    first_blocks, first_folds = build_temporal_design(times)
    second_blocks, second_folds = build_temporal_design(times.copy())

    assert [
        _sample_identity(times[block.positions]) for block in first_blocks
    ] == [
        _sample_identity(times[block.positions]) for block in second_blocks
    ]
    assert all(
        np.array_equal(first.outer_pool_positions, second.outer_pool_positions)
        for first, second in zip(first_folds, second_folds, strict=True)
    )


def _aggregate(*, positive: bool) -> dict[str, dict[str, float]]:
    value = 0.01 if positive else -0.01
    result = {
        name: {
            "mean": value,
            "median": value,
            "standard_deviation": 0.0,
            "minimum": value,
            "maximum": value,
        }
        for name in PRIMARY_METRIC_PATHS
    }
    result["directional_accuracy"] = {
        "mean": 0.51 if positive else 0.49,
        "median": 0.51 if positive else 0.49,
        "standard_deviation": 0.0,
        "minimum": 0.51 if positive else 0.49,
        "maximum": 0.51 if positive else 0.49,
    }
    return result


def test_temporal_stability_gate_is_exact() -> None:
    folds = [_aggregate(positive=True) for _ in range(3)] + [
        _aggregate(positive=False)
    ]
    positive, stable = temporal_stability_gate(folds, _aggregate(positive=True))

    assert positive == [True, True, True, False]
    assert stable is True

    _, unstable = temporal_stability_gate(
        [_aggregate(positive=True) for _ in range(2)]
        + [_aggregate(positive=False) for _ in range(2)],
        _aggregate(positive=True),
    )
    assert unstable is False


def test_overall_aggregation_uses_all_evaluations() -> None:
    evaluations = []
    for value in (0.01, 0.02):
        evaluation = {
            path[0]: {} for path in PRIMARY_METRIC_PATHS.values()
        }
        for path in PRIMARY_METRIC_PATHS.values():
            evaluation[path[0]][path[1]] = value
        evaluations.append(evaluation)

    aggregate = aggregate_evaluations(evaluations)

    assert aggregate["mae_skill"]["mean"] == pytest.approx(0.015)
    assert aggregate["mae_skill"]["standard_deviation"] == pytest.approx(0.005)
