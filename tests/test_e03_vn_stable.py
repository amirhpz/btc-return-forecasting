from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES
from btc_forecasting.features.f0 import F0_FEATURE_NAMES
from btc_forecasting.training.lstm import _sample_identity
from btc_forecasting.training.lstm_vn_mse import resolve_vn_mse_configuration
from btc_forecasting.training.lstm_vn_stable import (
    TARGET_COLUMN,
    _load_development_targets,
    build_common_paired_split,
    detect_f0_cross_redundancy,
    resolve_s9_configuration,
)


def _split_metadata() -> dict[str, object]:
    return {
        "split": {
            "retained_rows": {"train": 1, "validation": 1},
            "boundaries": {
                "train": {
                    "decision_time_start_inclusive": "2020-01-01T00:00:00Z",
                    "target_time_end_exclusive": "2020-02-01T00:00:00Z",
                },
                "validation": {
                    "decision_time_start_inclusive": "2020-02-01T00:00:00Z",
                    "target_time_end_exclusive": "2020-03-01T00:00:00Z",
                },
                "test": {
                    "decision_time_start_inclusive": "2020-03-01T00:00:00Z",
                },
            },
        }
    }


def test_cross_redundancy_uses_feature_values_without_target_input() -> None:
    times = pd.date_range("2020-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(-2.0, 3.0, len(times))
    f0 = pd.DataFrame({"open_time": times})
    for index, name in enumerate(F0_FEATURE_NAMES, 1):
        f0[name] = np.sin(base * index) + index * 0.01
    eng = pd.DataFrame({"open_time": times})
    for index, name in enumerate(ENG52_FEATURE_NAMES, 1):
        eng[name] = np.cos(base * (index + 11))
    duplicate = "band_bb_percB_20_2"
    eng[duplicate] = f0["log_return_1h"]

    report, additions = detect_f0_cross_redundancy(
        f0,
        eng,
        times,
        stable_candidates=(duplicate,),
    )

    assert additions == ()
    exclusion = report["excluded_eng_features"][0]  # type: ignore[index]
    assert exclusion["excluded_eng_feature"] == duplicate
    assert exclusion["matching_f0_feature"] == "log_return_1h"
    assert exclusion["exact_equivalence"] is True
    assert report["scope"] == (
        "frozen TRAIN feature values only; target values are not inputs"
    )


def test_common_builder_gives_both_models_identical_train_and_validation_identities() -> None:
    times = pd.date_range("2021-01-01", periods=32, freq="h", tz="UTC")
    features = pd.DataFrame({"open_time": times})
    for index, name in enumerate(F0_FEATURE_NAMES, 1):
        features[name] = np.arange(len(times), dtype=float) + index
    features["rolling_volatility_24h"] = 2.0
    additions = ("band_bb_percB_20_2", "roc_10")
    features[additions[0]] = np.linspace(0.0, 1.0, len(times))
    features[additions[1]] = np.linspace(1.0, 0.0, len(times))
    anchors = times[23:]
    targets = pd.DataFrame(
        {
            "bar_open_time": anchors,
            "decision_time": anchors + timedelta(hours=1),
            "target_time": anchors + timedelta(hours=2),
            TARGET_COLUMN: np.full(len(anchors), 0.4),
        }
    )

    train = build_common_paired_split(features, targets, eng_additions=additions)
    validation = build_common_paired_split(
        features,
        targets.iloc[2:].reset_index(drop=True),
        eng_additions=additions,
    )

    assert train.control.features.shape == (9, 24, 10)
    assert train.candidate.features.shape == (9, 24, 12)
    for paired in (train, validation):
        assert _sample_identity(paired.control.decision_times) == _sample_identity(
            paired.candidate.decision_times
        )
        assert np.array_equal(paired.control.targets, paired.candidate.targets)


def test_development_loader_filters_test_at_parquet_read(monkeypatch) -> None:
    rows = pd.DataFrame(
        {
            "bar_open_time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 2, 1, tzinfo=UTC),
            ],
            "decision_time": [
                datetime(2020, 1, 1, 1, tzinfo=UTC),
                datetime(2020, 2, 1, 1, tzinfo=UTC),
            ],
            "target_time": [
                datetime(2020, 1, 1, 2, tzinfo=UTC),
                datetime(2020, 2, 1, 2, tzinfo=UTC),
            ],
            TARGET_COLUMN: [0.01, -0.02],
        }
    )
    captured: dict[str, object] = {}

    class FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return rows.copy()

    def fake_read_table(path: Path, *, columns: list[str], filters: list[tuple]) -> FakeTable:
        captured.update(path=path, columns=columns, filters=filters)
        return FakeTable()

    monkeypatch.setattr(
        "btc_forecasting.training.lstm_vn_stable.pq.read_table", fake_read_table
    )
    train, validation = _load_development_targets(
        Path("targets.parquet"),
        split_metadata=_split_metadata(),
    )

    assert len(train) == 1
    assert len(validation) == 1
    assert captured["filters"] == [
        ("decision_time", "<", datetime(2020, 3, 1, tzinfo=UTC))
    ]


def test_s9_preserves_frozen_vn_mse_training_semantics() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = resolve_vn_mse_configuration(project_root=root)
    resolved = resolve_s9_configuration(project_root=root)

    assert resolved.baseline.model == baseline.model
    assert resolved.baseline.training == baseline.training
    assert resolved.baseline.training["loss"] == {
        "type": "torch.nn.MSELoss",
        "reduction": "mean",
        "target_scale": "causal_rolling_volatility_24h_normalized_return",
    }
    assert resolved.baseline.experiment["seed"] == 42
