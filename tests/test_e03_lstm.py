from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from btc_forecasting.common.config import load_yaml
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.models.lstm import LSTMRegressor
from btc_forecasting.targets.one_hour import TARGET_COLUMN
from btc_forecasting.training.lstm import (
    SequenceSamples,
    build_sequence_samples,
    build_training_components,
    evaluate_validation,
    fit_train_feature_scaler,
    require_official_cuda,
)


def _samples(features: np.ndarray, targets: np.ndarray, *, start: str) -> SequenceSamples:
    return SequenceSamples(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float64),
        decision_times=pd.date_range(start, periods=len(targets), freq="h", tz="UTC"),
        candidate_count=len(targets),
        excluded_lookback_count=0,
    )


def test_lstm_accepts_24_by_10_sequences_and_returns_one_scalar_per_sample() -> None:
    model = LSTMRegressor(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        configured_dropout=0.20,
    )
    output = model(torch.zeros((4, 24, 10), dtype=torch.float32))

    assert output.shape == (4,)
    assert model.configured_dropout == 0.20
    assert model.effective_lstm_dropout == 0.0
    assert model.lstm.dropout == 0.0
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())


def test_lstm_readout_uses_final_top_layer_hidden_state() -> None:
    class StubLSTM(torch.nn.Module):
        def forward(
            self, features: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            batch_size = features.shape[0]
            hidden = torch.full((1, batch_size, 2), 3.0)
            cell = torch.zeros_like(hidden)
            return torch.zeros((batch_size, 24, 2)), (hidden, cell)

    model = LSTMRegressor(
        input_size=10,
        hidden_size=2,
        num_layers=1,
        configured_dropout=0.20,
    )
    model.lstm = StubLSTM()
    with torch.no_grad():
        model.regression_head.weight.fill_(1.0)
        model.regression_head.bias.zero_()

    output = model(torch.zeros((3, 24, 10), dtype=torch.float32))

    assert output.tolist() == [6.0, 6.0, 6.0]


def test_e03_sequences_never_cross_an_hourly_gap() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    hours = [hour for hour in range(35) if hour != 10]
    open_times = pd.DatetimeIndex([start + timedelta(hours=hour) for hour in hours])
    feature_rows = pd.DataFrame({"open_time": open_times})
    for feature_index, name in enumerate(F0_FEATURE_NAMES):
        feature_rows[name] = np.arange(len(feature_rows), dtype=float) + feature_index
    anchors = pd.DatetimeIndex([start + timedelta(hours=24), start + timedelta(hours=34)])
    targets = pd.DataFrame(
        {
            "bar_open_time": anchors,
            "decision_time": anchors + timedelta(hours=1),
            "target_time": anchors + timedelta(hours=2),
            TARGET_COLUMN: [0.1, 0.2],
        }
    )

    samples = build_sequence_samples(feature_rows, targets)

    assert samples.candidate_count == 2
    assert samples.excluded_lookback_count == 1
    assert samples.features.shape == (1, 24, 10)
    assert samples.targets.tolist() == [0.2]


def test_feature_scaler_is_fit_from_train_timesteps_only() -> None:
    train_features = np.arange(2 * 24 * 10, dtype=np.float32).reshape(2, 24, 10)
    validation_features = train_features[:1] + 1_000_000.0
    train = _samples(train_features, np.array([0.1, 0.2]), start="2026-01-01")
    validation = _samples(validation_features, np.array([0.3]), start="2026-02-01")

    scaled = fit_train_feature_scaler(train, validation)

    expected_center = np.median(train_features.reshape(-1, 10), axis=0)
    assert np.array_equal(scaled.feature_scaler.center_, expected_center)
    assert scaled.train.features.shape == (2, 24, 10)
    assert scaled.validation.features.shape == (1, 24, 10)


def test_validation_evaluation_uses_same_rows_and_never_accepts_test_data() -> None:
    train = _samples(np.zeros((2, 24, 10)), np.array([0.1, -0.1]), start="2026-01-01")
    validation = _samples(
        np.zeros((3, 24, 10)),
        np.array([-0.2, 0.1, 0.3]),
        start="2026-02-01",
    )

    result = evaluate_validation(
        validation=validation,
        predictions=np.array([-0.1, 0.05, 0.2]),
        train=train,
    )

    assert result["evaluated_splits"] == ["validation"]
    assert result["lstm_validation"]["n"] == 3
    assert result["zero_return_same_validation_rows"]["n"] == 3
    assert result["lstm_validation"]["sample_identity_sha256"] == result[
        "zero_return_same_validation_rows"
    ]["sample_identity_sha256"]
    assert result["test_set"] == "NOT EVALUATED"
    assert "test" not in result


def test_official_execution_fails_instead_of_falling_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        require_official_cuda()


def test_frozen_loss_optimizer_and_scheduler_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    training = load_yaml(root / "configs" / "training.yaml")["training"]
    model = LSTMRegressor(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        configured_dropout=0.20,
    )

    optimizer, loss, scheduler = build_training_components(
        model,
        training_config=training,
    )

    assert isinstance(loss, torch.nn.HuberLoss)
    assert loss.delta == 0.01
    assert loss.reduction == "mean"
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["amsgrad"] is False
    assert scheduler.T_max == 30
    assert scheduler.eta_min == 0.0
