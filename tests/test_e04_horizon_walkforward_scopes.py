from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import SequenceSamples, TrainingOutcome
from btc_forecasting.training.lstm_horizon_walkforward import (
    PairedHorizonSamples,
    train_horizon_fold_seed,
)
from btc_forecasting.training.lstm_vn_walkforward import FixedEpochOutcome, OuterFold


def _samples() -> PairedHorizonSamples:
    count = 30
    times = pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC")
    raw = {
        horizon: np.linspace(-0.02 * horizon, 0.03 * horizon, count)
        for horizon in (1, 3, 6, 12)
    }
    sigma = np.full(count, 0.01)
    return PairedHorizonSamples(
        features=np.arange(count * 24 * len(F0_FEATURE_NAMES), dtype=np.float32).reshape(
            count, 24, len(F0_FEATURE_NAMES)
        ),
        decision_times=times,
        sigma=sigma,
        raw_targets=raw,
        normalized_targets={horizon: values / sigma for horizon, values in raw.items()},
    )


def test_inner_and_outer_learned_state_use_only_their_frozen_training_scopes(
    monkeypatch,
) -> None:
    samples = _samples()
    fold = OuterFold(
        number=1,
        outer_pool_positions=np.arange(20),
        inner_train_positions=np.arange(12),
        inner_validation_positions=np.arange(12, 16),
        outer_evaluation_positions=np.arange(20, 25),
        purged_outer_boundary_count=12,
        purged_inner_boundary_count=12,
    )
    events: list[tuple[str, pd.DatetimeIndex]] = []
    scale_scopes: list[np.ndarray] = []

    def fake_scales(targets, positions):
        scale_scopes.append(positions.copy())
        scale = 2.0 if len(scale_scopes) == 1 else 3.0
        return {1: 1.0, 3: scale, 6: scale + 1.0, 12: scale + 2.0}

    def fake_fit_scaler(sequence: SequenceSamples):
        events.append(("fit_scaler", sequence.decision_times))
        return object(), sequence

    def fake_transform(sequence: SequenceSamples, scaler: object):
        events.append(("transform", sequence.decision_times))
        return sequence

    def fake_fit_lstm(model, *, train, validation, training_config, seed, device):
        events.append(("inner_train", train.decision_times))
        events.append(("inner_validation", validation.decision_times))
        return TrainingOutcome(
            model=model,
            history=[{"epoch": 1}],
            best_epoch=2,
            epochs_trained=3,
            best_validation_loss=0.1,
            duration_seconds=1.0,
        )

    def fake_refit(model, *, train, training_config, seed, device, epochs):
        assert epochs == 2
        events.append(("outer_refit", train.decision_times))
        return FixedEpochOutcome(
            model=model,
            history=[{"epoch": 1}, {"epoch": 2}],
            duration_seconds=2.0,
        )

    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.relative_horizon_scales",
        fake_scales,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.configure_determinism",
        lambda seed: None,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward._new_model",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.fit_scope_scaler",
        fake_fit_scaler,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.transform_scope",
        fake_transform,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.fit_lstm",
        fake_fit_lstm,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.fit_lstm_fixed_epochs",
        fake_refit,
    )
    monkeypatch.setattr(
        "btc_forecasting.training.lstm_horizon_walkforward.predict_lstm",
        lambda model, features, batch_size, device: np.linspace(-0.1, 0.2, len(features)),
    )

    trained = train_horizon_fold_seed(
        horizon_hours=3,
        fold=fold,
        seed=42,
        samples=samples,
        model_config={"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        training_config={"max_epochs": 30, "batch_size": 128},
        device=torch.device("cpu"),
    )

    assert np.array_equal(scale_scopes[0], fold.inner_train_positions)
    assert np.array_equal(scale_scopes[1], fold.outer_pool_positions)
    assert [name for name, _ in events] == [
        "fit_scaler",
        "transform",
        "inner_train",
        "inner_validation",
        "fit_scaler",
        "outer_refit",
        "transform",
    ]
    assert events[0][1].equals(samples.decision_times[fold.inner_train_positions])
    assert events[1][1].equals(samples.decision_times[fold.inner_validation_positions])
    assert events[-2][1].equals(samples.decision_times[fold.outer_pool_positions])
    assert events[-1][1].equals(samples.decision_times[fold.outer_evaluation_positions])
    assert trained.metrics["training"]["inner_c_h"] == 2.0  # type: ignore[index]
    assert trained.metrics["training"]["outer_c_h"] == 3.0  # type: ignore[index]
