from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_forecasting.targets.horizon_audit import FROZEN_HORIZONS_HOURS, _sample_identity
from btc_forecasting.training.lstm_vn_multiseed import FROZEN_SEEDS, PRIMARY_METRIC_PATHS
from btc_forecasting.training.lstm_horizon_walkforward import (
    EXPECTED_COMMON_COUNT,
    EXPECTED_COMMON_IDENTITY,
    MAXIMUM_HORIZON_HOURS,
    build_paired_temporal_design,
    relative_horizon_scales,
    select_regression_horizon,
    validate_horizon_audit,
)


def test_frozen_horizons_seeds_and_e04_h_a_gate(tmp_path: Path) -> None:
    audit_path = tmp_path / "outputs" / "data" / "horizon_audit" / "summary.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "frozen_horizons_hours": [1, 3, 6, 12],
                "common_paired_anchors": {
                    "sample_count": EXPECTED_COMMON_COUNT,
                    "sample_identity_sha256": EXPECTED_COMMON_IDENTITY,
                },
                "scope": {
                    "original_validation": "NOT READ OR USED",
                    "test": "NOT READ OR USED",
                },
            }
        ),
        encoding="utf-8",
    )

    assert FROZEN_HORIZONS_HOURS == (1, 3, 6, 12)
    assert FROZEN_SEEDS == (42, 137, 271, 811, 2027)
    assert validate_horizon_audit(tmp_path)["common_paired_anchors"] == {
        "sample_count": 53_013,
        "sample_identity_sha256": EXPECTED_COMMON_IDENTITY,
    }


def test_six_paired_blocks_four_expanding_folds_and_max_horizon_purge() -> None:
    times = pd.date_range("2020-01-01", periods=121, freq="h", tz="UTC")
    blocks, folds = build_paired_temporal_design(times)

    assert len(blocks) == 6
    assert len(folds) == 4
    assert np.array_equal(np.concatenate([block.positions for block in blocks]), np.arange(121))
    assert max(map(lambda block: len(block.positions), blocks)) - min(
        map(lambda block: len(block.positions), blocks)
    ) == 1
    assert MAXIMUM_HORIZON_HOURS == 12
    for fold, expected_evaluation in zip(folds, blocks[2:], strict=True):
        assert np.array_equal(fold.outer_evaluation_positions, expected_evaluation.positions)
        inner_boundary = times[fold.inner_validation_positions[0]]
        outer_boundary = times[fold.outer_evaluation_positions[0]]
        assert times[fold.inner_train_positions[-1]] + pd.Timedelta(hours=12) < inner_boundary
        assert times[fold.outer_pool_positions[-1]] + pd.Timedelta(hours=12) < outer_boundary
        for horizon in FROZEN_HORIZONS_HOURS:
            assert np.all(times[fold.inner_train_positions] + pd.Timedelta(hours=horizon) < inner_boundary)
            assert np.all(times[fold.outer_pool_positions] + pd.Timedelta(hours=horizon) < outer_boundary)


def test_all_horizons_share_identical_paired_decisions() -> None:
    times = pd.date_range("2020-01-01", periods=60, freq="h", tz="UTC")
    blocks, folds = build_paired_temporal_design(times)
    identities = {
        horizon: [
            _sample_identity(times[fold.outer_evaluation_positions]) for fold in folds
        ]
        for horizon in FROZEN_HORIZONS_HOURS
    }

    assert all(value == identities[1] for value in identities.values())
    assert [len(block.positions) for block in blocks] == [10] * 6


def test_relative_scale_is_training_scope_only_with_exact_one_hour_control() -> None:
    q = {
        1: np.array([1.0, 1.0, 10.0]),
        3: np.array([2.0, 2.0, 90.0]),
        6: np.array([3.0, 3.0, 160.0]),
        12: np.array([4.0, 4.0, 250.0]),
    }
    train = np.array([0, 1])
    scales = relative_horizon_scales(q, train)

    assert scales == {1: 1.0, 3: 2.0, 6: 3.0, 12: 4.0}
    assert scales[12] != pytest.approx(np.sqrt(12))
    changed_evaluation = {horizon: values.copy() for horizon, values in q.items()}
    for values in changed_evaluation.values():
        values[2] *= 1_000.0
    assert relative_horizon_scales(changed_evaluation, train) == scales


def _aggregate(
    *,
    value: float,
    rmse_skill: float | None = None,
    mae_skill: float | None = None,
    r2: float | None = None,
) -> dict[str, dict[str, float]]:
    aggregate = {
        name: {
            "mean": value,
            "median": value,
            "standard_deviation": 0.0,
            "minimum": value,
            "maximum": value,
        }
        for name in PRIMARY_METRIC_PATHS
    }
    aggregate["rmse_skill"]["mean"] = value if rmse_skill is None else rmse_skill
    aggregate["mae_skill"]["mean"] = value if mae_skill is None else mae_skill
    aggregate["r2"]["mean"] = value if r2 is None else r2
    return aggregate


def _horizon_result(
    *,
    stable: bool,
    positive_folds: int,
    rmse_skill: float,
    mae_skill: float,
    r2: float,
) -> dict[str, object]:
    return {
        "HORIZON_REGRESSION_STABLE": stable,
        "regression_positive_fold_count": positive_folds,
        "overall_20_evaluation_aggregate": _aggregate(
            value=0.01,
            rmse_skill=rmse_skill,
            mae_skill=mae_skill,
            r2=r2,
        ),
    }


def test_horizon_selection_uses_only_stable_horizons_and_frozen_ranking() -> None:
    none = {
        horizon: _horizon_result(
            stable=False,
            positive_folds=4,
            rmse_skill=1.0,
            mae_skill=1.0,
            r2=1.0,
        )
        for horizon in FROZEN_HORIZONS_HOURS
    }
    assert select_regression_horizon(none) is None

    ranked = dict(none)
    ranked[1] = _horizon_result(
        stable=True,
        positive_folds=3,
        rmse_skill=0.03,
        mae_skill=0.03,
        r2=0.03,
    )
    ranked[3] = _horizon_result(
        stable=True,
        positive_folds=4,
        rmse_skill=0.01,
        mae_skill=0.01,
        r2=0.01,
    )
    assert select_regression_horizon(ranked) == 3

    ranked[1] = _horizon_result(
        stable=True,
        positive_folds=4,
        rmse_skill=0.02,
        mae_skill=0.01,
        r2=0.01,
    )
    assert select_regression_horizon(ranked) == 1
