from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecasting.features.eng52 import ENG52_FEATURE_NAMES
from btc_forecasting.features.eng52_screening import (
    BLOCK_COUNT,
    STABLE_CANDIDATE,
    WEAK_OR_UNSTABLE,
    ScreeningSamples,
    _load_train_targets,
    build_screening_samples,
    chronological_blocks,
    classify_spearman_signal,
    preferred_representative,
    screen_features,
)


def _split_metadata(*, train_rows: int = 1) -> dict[str, object]:
    return {
        "split": {
            "retained_rows": {"train": train_rows},
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


def test_target_loader_physically_requests_frozen_train_only(monkeypatch) -> None:
    captured: dict[str, object] = {}
    row = pd.DataFrame(
        {
            "bar_open_time": [datetime(2020, 1, 1, tzinfo=UTC)],
            "decision_time": [datetime(2020, 1, 1, 1, tzinfo=UTC)],
            "target_time": [datetime(2020, 1, 1, 2, tzinfo=UTC)],
            "future_log_return_1h": [0.01],
        }
    )

    class FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return row.copy()

    def fake_read_table(path: Path, *, columns: list[str], filters: list[tuple]) -> FakeTable:
        captured.update(path=path, columns=columns, filters=filters)
        return FakeTable()

    monkeypatch.setattr(
        "btc_forecasting.features.eng52_screening.pq.read_table", fake_read_table
    )
    result = _load_train_targets(Path("targets.parquet"), split_metadata=_split_metadata())

    assert len(result) == 1
    assert captured["filters"] == [
        ("decision_time", ">=", datetime(2020, 1, 1, tzinfo=UTC)),
        ("decision_time", "<", datetime(2020, 2, 1, tzinfo=UTC)),
    ]


def test_build_screening_samples_uses_gap_safe_24h_windows_and_endpoint_sigma() -> None:
    times = pd.date_range("2021-01-01", periods=30, freq="h", tz="UTC")
    eng52 = pd.DataFrame({"open_time": times})
    for index, name in enumerate(ENG52_FEATURE_NAMES, 1):
        eng52[name] = np.arange(30, dtype=float) + index
    f0 = pd.DataFrame(
        {"open_time": times, "rolling_volatility_24h": np.full(30, 2.0)}
    )
    anchors = times[23:]
    targets = pd.DataFrame(
        {
            "bar_open_time": anchors,
            "decision_time": anchors + timedelta(hours=1),
            "target_time": anchors + timedelta(hours=2),
            "future_log_return_1h": np.full(len(anchors), 0.5),
        }
    )

    samples = build_screening_samples(eng52, f0, targets)

    assert len(samples.normalized_targets) == 7
    assert np.array_equal(samples.normalized_targets, np.full(7, 0.25))
    assert samples.endpoint_features.iloc[0, 0] == eng52.iloc[23, 1]
    assert samples.excluded_incomplete_window == 0


def test_exactly_six_contiguous_chronological_blocks() -> None:
    blocks = chronological_blocks(62)

    assert len(blocks) == BLOCK_COUNT
    assert np.array_equal(np.concatenate(blocks), np.arange(62))
    assert max(map(len, blocks)) - min(map(len, blocks)) == 1


def test_frozen_candidate_rule_is_applied_at_exact_thresholds() -> None:
    stable, summary = classify_spearman_signal(
        0.02,
        [0.02, 0.02, 0.02, 0.02, 0.02, -0.01],
    )
    weak_full, _ = classify_spearman_signal(
        0.019,
        [0.03, 0.03, 0.03, 0.03, 0.03, -0.01],
    )
    weak_signs, _ = classify_spearman_signal(
        0.03,
        [0.03, 0.03, 0.03, 0.03, -0.03, -0.03],
    )

    assert stable == STABLE_CANDIDATE
    assert summary["sign_consistent_block_count"] == 5
    assert summary["median_absolute_spearman"] == 0.02
    assert weak_full == WEAK_OR_UNSTABLE
    assert weak_signs == WEAK_OR_UNSTABLE


def test_redundancy_tie_break_is_deterministic() -> None:
    def report(classification: str) -> dict[str, object]:
        return {
            "screening_classification": classification,
            "full_train": {"spearman": 0.04},
            "spearman_stability": {"median_absolute_spearman": 0.03},
            "train_endpoint_missing_ratio": 0.1,
        }

    reports = {
        "zeta": report(STABLE_CANDIDATE),
        "alpha": report(STABLE_CANDIDATE),
        "weak": report(WEAK_OR_UNSTABLE),
    }

    assert preferred_representative("zeta", "alpha", reports) == ("alpha", "zeta")
    assert preferred_representative("weak", "zeta", reports) == ("zeta", "weak")


def test_all_52_features_receive_a_screening_classification() -> None:
    sample_count = 60
    target = np.linspace(-1.0, 1.0, sample_count)
    features = pd.DataFrame(
        {
            name: target + (index + 1) * 1e-5 * np.sin(np.arange(sample_count))
            for index, name in enumerate(ENG52_FEATURE_NAMES)
        }
    )
    samples = ScreeningSamples(
        endpoint_features=features,
        normalized_targets=target,
        decision_times=pd.date_range(
            "2022-01-01", periods=sample_count, freq="h", tz="UTC"
        ),
        candidate_count=sample_count,
        excluded_incomplete_window=0,
        excluded_nonfinite_eng52_window=0,
        excluded_invalid_sigma=0,
        endpoint_missing_ratios={name: 0.0 for name in ENG52_FEATURE_NAMES},
    )

    reports, blocks = screen_features(samples)

    assert len(blocks) == 6
    assert tuple(reports) == ENG52_FEATURE_NAMES
    assert all(
        report["screening_classification"] in {STABLE_CANDIDATE, WEAK_OR_UNSTABLE}
        for report in reports.values()
    )
